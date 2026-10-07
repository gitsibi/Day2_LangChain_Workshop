import ast
import json
import math

import streamlit as st
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Code Complexity Coach", page_icon="🧠", layout="wide")
st.title("🧠 Code Complexity Coach (DSA)")
st.caption("Paste a code snippet → get complexity, a simple explanation, and a better approach.")


# ---------------- CUSTOM TOOLS (@tool) ----------------
def _loop_depth(node, depth=0):
    """Max nesting depth of loops (for / while / comprehensions)."""
    best = depth
    for child in ast.iter_child_nodes(node):
        add = 0
        if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            add = 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            add = len(child.generators)
        best = max(best, _loop_depth(child, depth + add))
    return best


@tool
def static_code_scan(code: str) -> str:
    """Statically scan Python code and report loop nesting depth, recursion,
    sorting calls and extra data structures. Returns a JSON string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return json.dumps({"error": f"Could not parse as Python: {e}"})

    recursive = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call)]:
            if isinstance(call.func, ast.Name) and call.func.id == fn.name:
                recursive.append(fn.name)
                break

    uses_sort = any(
        isinstance(c, ast.Call)
        and (
            (isinstance(c.func, ast.Name) and c.func.id == "sorted")
            or (isinstance(c.func, ast.Attribute) and c.func.attr == "sort")
        )
        for c in ast.walk(tree)
    )
    extra_structs = sorted({
        type(n).__name__
        for n in ast.walk(tree)
        if isinstance(n, (ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
    })

    return json.dumps({
        "max_loop_nesting_depth": _loop_depth(tree),
        "recursive_functions": recursive,
        "uses_sorting": uses_sort,
        "extra_data_structures": extra_structs,
    })


@tool
def estimate_operations(complexity: str, n: int = 1000) -> str:
    """Estimate how many operations an algorithm does for input size n,
    given a Big-O string like 'O(n^2)' or 'O(n log n)'."""
    c = (
        complexity.lower()
        .replace(" ", "")
        .replace("**", "^")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("o(", "")
        .rstrip(")")
    )
    funcs = {
        "n!": math.factorial,
        "2^n": lambda x: 2 ** x,
        "n^3": lambda x: x ** 3,
        "n^2": lambda x: x ** 2,
        "nlogn": lambda x: int(x * math.log2(max(x, 2))),
        "logn": lambda x: int(math.log2(max(x, 2))),
        "n": lambda x: x,
        "1": lambda x: 1,
    }
    key = next((k for k in funcs if c == k), None)
    if key is None:
        return f"Could not recognise '{complexity}'."

    sizes = [10, 100, n] if n not in (10, 100) else [10, 100, 1000]
    if key in ("n!", "2^n"):
        sizes = [5, 10, 20]  # keep numbers sane
    lines = []
    for s in sizes:
        ops = funcs[key](s)
        if ops < 10**15:
            lines.append(f"n={s:,} → ~{ops:,} operations")
        else:
            lines.append(f"n={s:,} → ~10^{len(str(ops)) - 1} operations")
    return "\n".join(lines)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:  # no secrets file locally -> ask in the sidebar
        api_key = ""
    api_key = api_key or st.text_input("Google API Key", type="password")
    model_name = st.text_input("Model", value="gemini-3.8-flash")
    level = st.selectbox("Explain like I'm a…", ["Beginner", "College student", "Interviewer-ready"])
    n_size = st.number_input("Input size n (for estimator)", 10, 1_000_000, 1000)
    st.divider()
    st.subheader("🕘 Memory (past analyses)")
    if "history" not in st.session_state:
        st.session_state.history = []
    for i, h in enumerate(reversed(st.session_state.history[-5:]), 1):
        st.write(f"{i}. {h['time']} / {h['space']}")
    if st.button("Clear memory"):
        st.session_state.history = []
        st.session_state.pop("result", None)
        st.session_state.pop("chat", None)
        st.rerun()

if not api_key:
    st.info("Add your Google API key in the sidebar to begin.")
    st.stop()

llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)

# ---------------- CHAINS (prompt | llm | parser) ----------------
# Chain 1: code + tool scan -> time & space complexity (JSON)
chain1 = (
    ChatPromptTemplate.from_template(
        "You are a DSA expert. Analyse this code.\n\nCode:\n{code}\n\n"
        "Static scan results (from a tool):\n{scan}\n\n"
        "Reply ONLY with JSON: "
        '{{"time": "O(...)", "space": "O(...)", "reasoning": "one line"}}'
    )
    | llm
    | JsonOutputParser()
)

# Chain 2 (multi-input): simple explanation
chain2 = (
    ChatPromptTemplate.from_template(
        "Explain in simple words WHY this code has time complexity {time} and "
        "space complexity {space}. Audience: {level}. Use a short real-life analogy "
        "and point to the exact lines/loops responsible.\n\nCode:\n{code}"
    )
    | llm
    | StrOutputParser()
)

# Chain 3: better approach
chain3 = (
    ChatPromptTemplate.from_template(
        "Current code has time {time}, space {space}.\n"
        "Explanation so far:\n{explanation}\n\nCode:\n{code}\n\n"
        "Suggest a better approach (technique/data structure, e.g. hash map, "
        "two pointers, DP). Give the improved code, its new complexity, and the "
        "trade-off. If the code is already optimal, say so."
    )
    | llm
    | StrOutputParser()
)

# Follow-up chat with memory
followup = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a friendly DSA coach. Analysis context:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# ---------------- UI ----------------
SAMPLE = '''def has_duplicate(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False
'''
code = st.text_area("Paste your code (Python gives the best tool scan)", SAMPLE, height=250)

if st.button("🚀 Analyse", type="primary"):
    try:
        with st.status("Running the chains…", expanded=True) as status:
            st.write("🔧 Tool 1: static_code_scan")
            scan = static_code_scan.invoke({"code": code})

            st.write("⛓️ Chain 1: finding complexity")
            c1 = chain1.invoke({"code": code, "scan": scan})

            st.write("🔧 Tool 2: estimate_operations (fed by Chain 1)")
            ops = estimate_operations.invoke({"complexity": c1["time"], "n": int(n_size)})

            st.write("⛓️ Chain 2: simple explanation")
            explanation = chain2.invoke({
                "code": code, "time": c1["time"], "space": c1["space"], "level": level,
            })

            st.write("⛓️ Chain 3: better approach")
            better = chain3.invoke({
                "code": code, "time": c1["time"], "space": c1["space"],
                "explanation": explanation,
            })
            status.update(label="Done ✅", state="complete")

        st.session_state.result = dict(
            scan=scan, c1=c1, ops=ops, explanation=explanation, better=better,
            code=code, n=int(n_size),
        )
        st.session_state.chat = []
        st.session_state.history.append({"time": c1["time"], "space": c1["space"]})
    except Exception as e:
        st.error(f"Something went wrong: {e}")

r = st.session_state.get("result")
if r:
    col1, col2 = st.columns(2)
    col1.metric("⏱️ Time Complexity", r["c1"]["time"])
    col2.metric("💾 Space Complexity", r["c1"]["space"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔧 Tools", "💡 Why?", " Better approach", " Ask follow-ups", "Teaching use"]
    )
    with tab1:
        st.markdown("### Tool 1: `static_code_scan`")
        st.caption("Input: your code → Output: structural facts")
        st.code(r["code"], language="python")
        st.json(json.loads(r["scan"]))

        st.markdown("### Tool 2: `estimate_operations`")
        st.caption(f"Input: complexity from Chain 1 = {r['c1']['time']}, n = {r['n']}")
        st.code(r["ops"])
    with tab2:
        st.markdown(r["explanation"])
    with tab3:
        st.markdown(r["better"])
    with tab4:
        for role, msg in st.session_state.get("chat", []):
            st.chat_message("user" if role == "human" else "assistant").write(msg)
        q = st.chat_input("Ask about this analysis…")
        if q:
            ctx = (
                f"Code:\n{r['code']}\nTime: {r['c1']['time']}, Space: {r['c1']['space']}\n"
                f"Better approach:\n{r['better']}"
            )
            ans = followup.invoke({
                "context": ctx, "history": st.session_state.chat, "question": q,
            })
            st.session_state.chat += [("human", q), ("ai", ans)]
            st.rerun()
    with tab5:
        st.markdown("""
**How I'd use this in my DSA teaching**
- **Instant feedback:** students paste their solution and see its complexity without waiting for me.
- **Tool + LLM together:** the AST scan gives hard facts and the LLM explains them, so there are fewer wrong guesses.
- **Concrete numbers:** `estimate_operations` shows what O(n²) means at n = 1000 (about a million operations).
- **Brute force to optimal:** Chain 3 shows the better technique, like a hash map, two pointers or DP.
- **Follow-up questions:** the memory lets students ask "why not sorting?" in context.
""")