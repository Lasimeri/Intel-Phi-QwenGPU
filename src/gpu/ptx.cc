// See ptx.h.
#include "ptx.h"

#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace ptx {

namespace {

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, "ptx: %s\n", what.c_str());
    std::exit(2);
}

std::string strip(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

bool starts(const std::string &s, const char *p) { return s.compare(0, std::strlen(p), p) == 0; }

std::vector<std::string> split_dots(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == '.') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    out.push_back(cur);
    return out;
}

int64_t parse_int(std::string s) {
    s = strip(s);
    if (!s.empty() && s[0] == '+') s = s.substr(1);
    bool neg = !s.empty() && s[0] == '-';
    if (neg) s = s.substr(1);
    while (!s.empty() && (s.back() == 'U' || s.back() == 'u')) s.pop_back();
    int64_t v;
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) v = (int64_t)std::strtoull(s.c_str() + 2, nullptr, 16);
    else v = (int64_t)std::strtoull(s.c_str(), nullptr, 10);
    return neg ? -v : v;
}

bool is_sreg(const std::string &s) {
    static const char *names[] = {"%tid.x", "%tid.y", "%tid.z", "%ntid.x", "%ntid.y", "%ntid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z",
                                  "%nctaid.x", "%nctaid.y", "%nctaid.z", "%laneid", "%warpid"};
    for (const char *n : names)
        if (s == n) return true;
    return false;
}

bool is_imm(const std::string &s) {
    size_t i = 0;
    if (i < s.size() && s[i] == '-') i++;
    std::string body = s.substr(i);
    if (!body.empty() && (body.back() == 'U' || body.back() == 'u')) body.pop_back();
    if (body.empty()) return false;
    if (body.size() > 2 && body[0] == '0' && (body[1] == 'x' || body[1] == 'X')) {
        for (size_t k = 2; k < body.size(); k++)
            if (!std::isxdigit((unsigned char)body[k])) return false;
        return body.size() > 2;
    }
    for (char ch : body)
        if (!std::isdigit((unsigned char)ch)) return false;
    return true;
}

std::vector<std::string> split_args(const std::string &s) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char ch : s) {
        if (ch == '[' || ch == '{') depth++;
        else if (ch == ']' || ch == '}') depth--;
        if (ch == ',' && depth == 0) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!strip(cur).empty()) out.push_back(cur);
    return out;
}

}  // namespace

Operand parse_operand(const std::string &raw) {
    std::string s = strip(raw);
    Operand o;
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
        std::string inner;
        for (char ch : s.substr(1, s.size() - 2))
            if (ch != ' ') inner += ch;
        // base, then an optional +off / -off
        size_t k = 0;
        while (k < inner.size() && inner[k] != '+' && inner[k] != '-') k++;
        o.kind = Operand::MEM;
        o.base = std::make_shared<Operand>(parse_operand(inner.substr(0, k)));
        o.value = k < inner.size() ? parse_int(inner.substr(k)) : 0;
        return o;
    }
    if (s.find('|') != std::string::npos) {
        size_t k = s.find('|');
        o.kind = Operand::PAIR;
        o.base = std::make_shared<Operand>(parse_operand(s.substr(0, k)));
        o.second = std::make_shared<Operand>(parse_operand(s.substr(k + 1)));
        return o;
    }
    if (is_sreg(s)) {
        o.kind = Operand::SREG;
        o.name = s;
        return o;
    }
    if (!s.empty() && s[0] == '%') {
        o.kind = Operand::REG;
        o.name = s;
        return o;
    }
    if (!s.empty() && s[0] == '$') {
        o.kind = Operand::LABEL;
        o.name = s;
        return o;
    }
    if (is_imm(s)) {
        o.kind = Operand::IMM;
        o.value = parse_int(s);
        return o;
    }
    o.kind = Operand::SYM;
    o.name = s;
    return o;
}

// symbols inside an operand (a parameter's name in [name]) renamed by a table
static void rename_syms(Operand &o, const std::map<std::string, std::string> &rename) {
    if ((o.kind == Operand::SYM || o.kind == Operand::MEM) && rename.count(o.name)) o.name = rename.at(o.name);
    if (o.base) rename_syms(*o.base, rename);
    if (o.second) rename_syms(*o.second, rename);
}

std::map<std::string, Kernel> parse(const std::string &text) {
    std::map<std::string, Kernel> kernels;
    std::vector<std::string> lines;
    {
        std::istringstream in(text);
        std::string l;
        while (std::getline(in, l)) lines.push_back(l);
    }
    std::map<std::string, Kernel> funcs;  // .func definitions: clang's OpenCL front-end puts every kernel's body in one
    size_t i = 0;
    while (i < lines.size()) {
        std::string s = strip(lines[i]);
        // a kernel, or a function definition (a stub kernel takes its function's body below); a forward declaration has
        // no "(" on its header line and is skipped to its ";"; functions with a return value (.param before the name)
        // are never defined here
        const char *head = starts(s, ".visible .entry ") ? ".visible .entry " : starts(s, ".visible .func ") && s.find("(.param") == std::string::npos ? ".visible .func " : nullptr;
        if (!head) {
            i++;
            continue;
        }
        const bool is_func = head[10] == 'f';  /* ".visible .func " vs ".visible .entry " */
        if (s.find('(') == std::string::npos) {
            while (i < lines.size() && strip(lines[i]).back() != ';') i++;
            i++;
            continue;
        }
        Kernel k;
        bool no_params = false;
        {
            size_t a = std::strlen(head), b = s.find('(', a);
            k.name = strip(s.substr(a, b - a));
            no_params = s.find(')', b) != std::string::npos;  // `name()` on one line: nothing to read
        }
        i++;
        while (!no_params && i < lines.size() && !starts(strip(lines[i]), ")")) {
            std::string p = strip(lines[i]);
            if (starts(p, ".param ")) {
                // .param .u64 .ptr .align 1 name,  |  .param .u32 name
                std::istringstream ps(p.substr(7));
                std::string ty, tok, name;
                ps >> ty;
                while (ps >> tok) name = tok;
                if (!name.empty() && name.back() == ',') name.pop_back();
                Param pm;
                pm.name = name;
                pm.type = ty;
                pm.size = ty == ".u64" || ty == ".b64" || ty == ".s64" ? 8 : ty == ".u32" || ty == ".b32" || ty == ".s32" ? 4 : ty == ".u16" || ty == ".b16" ? 2 : 1;
                k.params.push_back(pm);
            }
            i++;
        }
        if (!no_params) i++;  // past the ")" line
        while (i < lines.size() && strip(lines[i]) != "{") i++;
        i++;
        while (i < lines.size() && strip(lines[i]) != "}") {
            std::string raw = lines[i];
            size_t c = raw.find("//");
            std::string st = strip(c == std::string::npos ? raw : raw.substr(0, c));
            i++;
            if (st.empty()) continue;
            if (st == "{" || st == "}" || starts(st, ".param ")) continue;  // a call sequence's scope and parameters
            if (starts(st, "call")) {                                      // the stub's call of its implementation
                size_t c1 = st.find(' '), c2 = st.find(',', c1);
                if (c1 == std::string::npos) die("line " + std::to_string(i) + ": " + st);
                k.stub_call = strip(st.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1));
                continue;
            }
            if (st.back() == ':') {
                k.labels[st.substr(0, st.size() - 1)] = (int)k.instrs.size();
                continue;
            }
            if (starts(st, ".reg ")) {
                // .reg .b32 %r<15>;  .reg .pred %p<6>;
                std::istringstream rs(st.substr(5));
                std::string ty, nm;
                rs >> ty >> nm;
                size_t lt = nm.find('<');
                std::string prefix = nm.substr(1, lt == std::string::npos ? std::string::npos : lt - 1);
                k.reg_width[prefix] = ty == ".pred" ? 1 : std::atoi(ty.c_str() + 2);
                continue;
            }
            if (starts(st, ".shared ")) {
                // .shared .align 4 .b8 name[128];
                std::istringstream ss(st.substr(8));
                std::string tok, align, ty, decl;
                ss >> tok >> align >> ty >> decl;
                Shared sh;
                sh.align = std::atoi(align.c_str());
                size_t lb = decl.find('['), rb = decl.find(']');
                sh.name = decl.substr(0, lb);
                sh.size = std::atoi(decl.substr(lb + 1, rb - lb - 1).c_str());
                k.shared.push_back(sh);
                continue;
            }
            if (starts(st, ".pragma")) continue;
            if (st[0] == '.') die("line " + std::to_string(i) + ": unsupported directive " + st);
            if (st.back() != ';') die("line " + std::to_string(i) + ": no ; in " + st);
            st = strip(st.substr(0, st.size() - 1));
            Instr ins;
            ins.line = (int)i;
            if (st[0] == '@') {
                size_t sp = st.find(' ');
                std::string g = st.substr(1, sp - 1);
                if (!g.empty() && g[0] == '!') {
                    ins.neg = true;
                    g = g.substr(1);
                }
                ins.guard = g;
                st = strip(st.substr(sp + 1));
            }
            size_t sp = st.find(' ');
            ins.op = sp == std::string::npos ? st : st.substr(0, sp);
            std::string rest = sp == std::string::npos ? "" : st.substr(sp + 1);
            if (!strip(rest).empty())
                for (const std::string &a : split_args(rest)) ins.args.push_back(parse_operand(a));
            ins.parts = split_dots(ins.op);
            k.instrs.push_back(ins);
        }
        for (const Instr &ins : k.instrs)
            for (const Operand &a : ins.args)
                if (a.kind == Operand::LABEL && !k.labels.count(a.name)) die(k.name + ": unknown label " + a.name);
        if (is_func) funcs[k.name] = k;
        else kernels[k.name] = k;
        i++;
    }
    // a stub kernel (only parameter moves and a call) takes its function's body; the function's parameters are the
    // kernel's, in order, so their names are rewritten
    for (auto &kv : kernels) {
        Kernel &k = kv.second;
        if (k.stub_call.empty()) continue;
        auto f = funcs.find(k.stub_call);
        if (f == funcs.end()) die(k.name + ": calls " + k.stub_call + ", which is not defined in this module");
        const Kernel &impl = f->second;
        for (const Instr &ins : k.instrs) {
            const bool param_move = (ins.base() == "ld" || ins.base() == "st") && ins.parts.size() > 1 && ins.parts[1] == "param";
            if (!param_move && ins.base() != "ret") die(k.name + ": a kernel that calls a function and does more: not supported");
        }
        if (impl.params.size() != k.params.size()) die(k.name + ": its implementation takes a different number of parameters");
        std::map<std::string, std::string> rename;
        for (size_t p = 0; p < k.params.size(); p++) rename[impl.params[p].name] = k.params[p].name;
        k.instrs = impl.instrs;
        k.labels = impl.labels;
        k.reg_width = impl.reg_width;
        k.shared = impl.shared;
        k.stub_call.clear();
        for (Instr &ins : k.instrs)
            for (Operand &a : ins.args) rename_syms(a, rename);
    }
    return kernels;
}

std::map<std::string, Kernel> load(const std::string &path) {
    std::ifstream f(path);
    if (!f) die("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

}  // namespace ptx
