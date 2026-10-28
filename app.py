import ast
import re
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


st.set_page_config(
    page_title="Intelligent UML-to-Code Compiler",
    page_icon="🧩",
    layout="wide",
)

# Beige, black and cream theme
st.markdown("""
<style>
:root {
    --ink: #29251f;
    --paper: #f3eadc;
    --sand: #e9dcc9;
    --line: #d4c2a8;
    --cream: #fffaf2;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
}
[data-testid="stHeader"] { background: rgba(243,234,220,.96) !important; }
[data-testid="stSidebar"] {
    background: var(--sand) !important;
    border-right: 1px solid var(--line);
}
h1, h2, h3, h4, p, label, li,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
}
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fbf7f0 !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px;
}
.stButton > button, .stDownloadButton > button {
    background: #29251f !important;
    color: #fffaf2 !important;
    border: 1px solid #29251f !important;
    border-radius: 9px;
}
.stButton > button *, .stDownloadButton > button * {
    color: #fffaf2 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #514537 !important;
    border-color: #514537 !important;
}
.stTextArea textarea, .stTextInput input,
[data-baseweb="select"] > div {
    background: #fffdf9 !important;
    color: #29251f !important;
    border-color: var(--line) !important;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #e9dcc9 !important;
    color: #29251f !important;
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] * { color: #29251f !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #29251f !important;
    color: #fffaf2 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] * {
    color: #fffaf2 !important;
}
hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)


LABELS = ["Factory", "Singleton", "Observer", "No clear pattern"]
FEATURE_COLUMNS = [
    "total_classes", "total_methods", "total_attributes",
    "factory_hint", "singleton_hint", "observer_hint",
]
TRAINING_COLUMNS = FEATURE_COLUMNS + ["label"]

TRAINING_ROWS = [
    [2, 4, 3, 1, 0, 0, "Factory"],
    [3, 5, 4, 1, 0, 0, "Factory"],
    [2, 6, 5, 1, 0, 0, "Factory"],
    [4, 7, 6, 1, 0, 0, "Factory"],
    [1, 3, 2, 0, 1, 0, "Singleton"],
    [2, 4, 3, 0, 1, 0, "Singleton"],
    [3, 5, 5, 0, 1, 0, "Singleton"],
    [4, 6, 7, 0, 1, 0, "Singleton"],
    [3, 6, 4, 0, 0, 1, "Observer"],
    [4, 8, 6, 0, 0, 1, "Observer"],
    [2, 5, 4, 0, 0, 1, "Observer"],
    [5, 9, 8, 0, 0, 1, "Observer"],
    [3, 6, 6, 0, 0, 0, "No clear pattern"],
    [2, 3, 3, 0, 0, 0, "No clear pattern"],
    [4, 7, 8, 0, 0, 0, "No clear pattern"],
]
DEFAULT_TRAINING_DF = pd.DataFrame(TRAINING_ROWS, columns=TRAINING_COLUMNS)

SAMPLE_DESIGNS = [
    """class User
attributes: name:str, email:str
methods: login(), logout()

class Order
attributes: order_id:int, total:float
methods: calculate_total(), place_order()

class Payment
attributes: amount:float, status:str
methods: pay(), refund()""",
    """class NotificationFactory
attributes: factory_name:str
methods: create_notification(), create_email(), create_sms()

class EmailNotification
attributes: recipient:str
methods: send(), format_message()""",
    """class AppConfig
attributes: instance:object, app_name:str
methods: get_instance(), load_config()""",
    """class EventPublisher
attributes: event_name:str
methods: subscribe(), notify(), update_subscribers()

class Listener
attributes: listener_id:int
methods: update(), handle_event()""",
    """class Library
attributes: name:str, address:str
methods: open(), close()

class Book
attributes: title:str, author:str
methods: borrow(), return_book()""",
]


@dataclass
class ClassInfo:
    class_name: str
    attributes: list = field(default_factory=list)
    methods: list = field(default_factory=list)


def parse_uml_model(source_text: str) -> list[ClassInfo]:
    class_list = []
    active_class = None

    for line_no, raw_line in enumerate(source_text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("class "):
            class_name = line[6:].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", class_name):
                raise ValueError(f"Invalid class name on line {line_no}.")
            if any(item.class_name == class_name for item in class_list):
                raise ValueError(f"Duplicate class name: {class_name}.")
            active_class = ClassInfo(class_name)
            class_list.append(active_class)

        elif line.startswith("attributes:"):
            if active_class is None:
                raise ValueError("Attributes must follow a class declaration.")
            text = line.split(":", 1)[1].strip()
            if text:
                for entry in text.split(","):
                    parts = entry.strip().split(":")
                    if len(parts) != 2:
                        raise ValueError(f"Invalid attribute on line {line_no}; use name:type.")
                    name, typ = [part.strip() for part in parts]
                    if not re.fullmatch(r"[A-Za-z_]\w*", name) or not re.fullmatch(r"[A-Za-z_]\w*", typ):
                        raise ValueError(f"Invalid attribute name or type on line {line_no}.")
                    active_class.attributes.append((name, typ))

        elif line.startswith("methods:"):
            if active_class is None:
                raise ValueError("Methods must follow a class declaration.")
            text = line.split(":", 1)[1].strip()
            if text:
                for entry in text.split(","):
                    match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(\s*\)\s*", entry)
                    if not match:
                        raise ValueError(f"Invalid method on line {line_no}; use method().")
                    active_class.methods.append(match.group(1))
        else:
            raise ValueError(f"Unsupported input on line {line_no}: {line}")

    if not class_list:
        raise ValueError("No classes found.")
    return class_list


def validate_design(classes):
    warnings = []
    if len(classes) > 10:
        warnings.append("The model has more than 10 classes.")

    for item in classes:
        if len(item.methods) > 8:
            warnings.append(f"{item.class_name} has more than 8 methods.")
        if len(item.attributes) > 8:
            warnings.append(f"{item.class_name} has more than 8 attributes.")
        if not item.methods:
            warnings.append(f"{item.class_name} has no methods.")
        if len(item.methods) != len(set(item.methods)):
            warnings.append(f"{item.class_name} has duplicate method names.")
        names = [name for name, _ in item.attributes]
        if len(names) != len(set(names)):
            warnings.append(f"{item.class_name} has duplicate attribute names.")
    return warnings


def extract_features(classes):
    methods = [method.lower() for item in classes for method in item.methods]
    attributes = [name.lower() for item in classes for name, _ in item.attributes]
    return [
        len(classes),
        sum(len(item.methods) for item in classes),
        sum(len(item.attributes) for item in classes),
        int(any(word in method for method in methods for word in ("create", "factory"))),
        int(any("instance" in attr or "singleton" in attr for attr in attributes)),
        int(any(word in method for method in methods for word in ("notify", "update", "subscribe"))),
    ]


def generate_python_code(classes):
    type_defaults = {
        "str": ("str", '""'),
        "string": ("str", '""'),
        "int": ("int", "0"),
        "float": ("float", "0.0"),
        "bool": ("bool", "False"),
        "boolean": ("bool", "False"),
    }
    lines = []
    for item in classes:
        lines.extend([f"class {item.class_name}:", "    def __init__(self):"])
        if not item.attributes:
            lines.append("        pass")
        for name, typ in item.attributes:
            py_type, default = type_defaults.get(typ.lower(), ("object", "None"))
            lines.append(f"        self.{name}: {py_type} = {default}")
        for method in item.methods:
            lines.extend(["", f"    def {method}(self):", f'        """Implement {method}."""', "        pass"])
        lines.append("")
    return "\n".join(lines)


def validate_training_df(df):
    missing = [column for column in TRAINING_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing training columns: " + ", ".join(missing))

    df = df[TRAINING_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
        if (df[column] < 0).any():
            raise ValueError(f"{column} must contain non-negative numbers.")

    for column in ["factory_hint", "singleton_hint", "observer_hint"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")

    df["label"] = df["label"].astype(str).str.strip()
    if not df["label"].isin(LABELS).all():
        raise ValueError("Invalid label. Use: " + ", ".join(LABELS))
    if len(df) < 8:
        raise ValueError("At least 8 training rows are required.")
    if df["label"].nunique() < 2:
        raise ValueError("At least two different labels are required.")
    return df


def train_model(df):
    X = df[FEATURE_COLUMNS].astype(int)
    y = df["label"].astype(str)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    train_accuracy = accuracy_score(y, model.predict(X))

    min_count = int(y.value_counts().min())
    folds = min(3, min_count)
    cv_scores = None
    cv_predictions = None

    if folds >= 2 and y.nunique() >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        evaluation_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        cv_scores = cross_val_score(evaluation_model, X, y, cv=cv, scoring="accuracy")
        cv_predictions = cross_val_predict(
            DecisionTreeClassifier(max_depth=3, random_state=42), X, y, cv=cv
        )
    return model, train_accuracy, cv_scores, cv_predictions


def _syntax_ok(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def analyze_design(source_text, model):
    started = time.perf_counter()
    classes = parse_uml_model(source_text)
    warnings = validate_design(classes)
    features = extract_features(classes)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = float(max(probabilities)) * 100
    code = generate_python_code(classes)
    ast.parse(code)

    return {
        "classes": classes,
        "warnings": warnings,
        "features": features,
        "prediction": prediction,
        "confidence": confidence,
        "code": code,
        "elapsed": (time.perf_counter() - started) * 1000,
        "attributes": sum(len(item.attributes) for item in classes),
        "methods": sum(len(item.methods) for item in classes),
        "source": source_text,
    }


# Sidebar data
with st.sidebar:
    st.header("Model")
    st.write("Decision Tree Classifier")
    st.write("Maximum depth: 3")
    st.write(f"Training examples: {len(TRAINING_ROWS)}")
    uploaded_csv = st.file_uploader("Training CSV", type=["csv"])
    st.download_button(
        "Download training data",
        DEFAULT_TRAINING_DF.to_csv(index=False).encode("utf-8"),
        "uml_training_data.csv",
        "text/csv",
        use_container_width=True,
    )

try:
    training_df = validate_training_df(pd.read_csv(uploaded_csv)) if uploaded_csv else DEFAULT_TRAINING_DF.copy()
    model, train_accuracy, cv_scores, cv_predictions = train_model(training_df)
except Exception as exc:
    st.error(f"Training data error: {exc}")
    st.stop()


# Session state
if "design_inputs" not in st.session_state:
    st.session_state.design_inputs = SAMPLE_DESIGNS.copy()
if "design_results" not in st.session_state:
    st.session_state.design_results = {}

for i, sample in enumerate(SAMPLE_DESIGNS, 1):
    if f"design_editor_{i}" not in st.session_state:
        st.session_state[f"design_editor_{i}"] = st.session_state.design_inputs[i - 1]


st.title("Intelligent UML-to-Code Compiler")

workspace_tab, performance_tab, data_tab = st.tabs([
    "Design workspace",
    "Model performance",
    "Training data",
])

with workspace_tab:
    design_names = [f"Design {i}" for i in range(1, 6)]
    selected_design = st.radio(
        "Design",
        design_names,
        horizontal=True,
        label_visibility="collapsed",
        key="selected_design",
    )
    design_no = int(selected_design.split()[-1])
    index = design_no - 1

    st.subheader(selected_design)

    def sync_design(no):
        st.session_state.design_inputs[no - 1] = st.session_state.get(f"design_editor_{no}", "")

    st.text_area(
        "UML input",
        key=f"design_editor_{design_no}",
        height=210,
        on_change=sync_design,
        args=(design_no,),
    )
    st.session_state.design_inputs[index] = st.session_state.get(
        f"design_editor_{design_no}", st.session_state.design_inputs[index]
    )

    button1, button2, button3 = st.columns([1, 1, 1])
    with button1:
        analyze_current = st.button("Analyze design", type="primary", use_container_width=True)
    with button2:
        analyze_all = st.button("Analyze all", use_container_width=True)
    with button3:
        clear_design = st.button("Clear design", use_container_width=True)

    if clear_design:
        st.session_state.design_inputs[index] = ""
        st.session_state[f"design_editor_{design_no}"] = ""
        st.session_state.design_results.pop(design_no, None)
        st.rerun()

    if analyze_current:
        source = st.session_state.design_inputs[index].strip()
        if not source:
            st.warning("The selected design is empty.")
        else:
            try:
                st.session_state.design_results[design_no] = analyze_design(source, model)
                st.success(f"{selected_design} complete.")
            except (ValueError, SyntaxError) as exc:
                st.error(str(exc))

    if analyze_all:
        done, errors = 0, []
        for number, source in enumerate(st.session_state.design_inputs, 1):
            source = source.strip()
            if not source:
                st.session_state.design_results.pop(number, None)
                continue
            try:
                st.session_state.design_results[number] = analyze_design(source, model)
                done += 1
            except (ValueError, SyntaxError) as exc:
                errors.append(f"Design {number}: {exc}")
        if done:
            st.success(f"{done} design(s) analyzed.")
        for error in errors:
            st.error(error)

    result = st.session_state.design_results.get(design_no)

    if result:
        st.divider()
        a, b, c, d = st.columns(4)
        a.metric("Classes", len(result["classes"]))
        b.metric("Attributes", result["attributes"])
        c.metric("Methods", result["methods"])
        d.metric("Processing time", f"{result['elapsed']:.2f} ms")

        st.subheader("Prediction")
        st.markdown(f"## {result['prediction']}")
        st.metric("Prediction probability", f"{result['confidence']:.1f}%")

        overview, structure, code_tab, features_tab = st.tabs([
            "Overview", "Class structure", "Python code", "Features"
        ])

        with overview:
            st.subheader("Validation")
            if result["warnings"]:
                for warning in result["warnings"]:
                    st.warning(warning)
            else:
                st.success("No structural warnings found.")
            st.write("Python syntax:", "Valid" if _syntax_ok(result["code"]) else "Invalid")

        with structure:
            for class_info in result["classes"]:
                with st.container(border=True):
                    st.markdown(f"### {class_info.class_name}")
                    left, right = st.columns(2)
                    left.write("**Attributes**")
                    left.write(", ".join(f"{n}: {t}" for n, t in class_info.attributes) or "None")
                    right.write("**Methods**")
                    right.write(", ".join(f"{m}()" for m in class_info.methods) or "None")

        with code_tab:
            st.code(result["code"], language="python")
            st.download_button(
                "Download Python file",
                result["code"],
                f"design_{design_no}.py",
                "text/x-python",
                key=f"code_download_{design_no}",
            )

        with features_tab:
            st.dataframe(
                pd.DataFrame({"Feature": FEATURE_COLUMNS, "Value": result["features"]}),
                use_container_width=True,
                hide_index=True,
            )

        report = (
            f"Design {design_no}\n"
            f"Prediction: {result['prediction']}\n"
            f"Prediction probability: {result['confidence']:.1f}%\n"
            f"Classes: {len(result['classes'])}\n"
            f"Attributes: {result['attributes']}\n"
            f"Methods: {result['methods']}\n"
            f"Features: {result['features']}\n"
            f"Warnings: {result['warnings'] or 'None'}\n\n"
            f"Generated Python:\n{result['code']}"
        )
        st.download_button(
            "Download report",
            report,
            f"design_{design_no}_report.txt",
            "text/plain",
            key=f"report_download_{design_no}",
        )


with performance_tab:
    st.subheader("Model performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training accuracy", f"{train_accuracy * 100:.1f}%")
    col2.metric("Training rows", len(training_df))
    if cv_scores is not None:
        col3.metric("Cross-validation accuracy", f"{cv_scores.mean() * 100:.1f}%")
    else:
        col3.metric("Cross-validation accuracy", "Unavailable")

    st.caption("The dataset contains 15 manually labelled examples. Results may not represent performance on new, real-world UML designs.")

    # Beige-and-black class distribution chart
    counts = training_df["label"].value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#f3eadc")
    ax.set_facecolor("#f3eadc")
    bars = ax.bar(counts.index, counts.values, color="#29251f", edgecolor="#fffaf2")
    ax.set_title("Training examples by category", color="#29251f")
    ax.set_ylabel("Number of examples", color="#29251f")
    ax.tick_params(axis="x", labelrotation=15, colors="#29251f")
    ax.tick_params(axis="y", colors="#29251f")
    ax.spines["bottom"].set_color("#d4c2a8")
    ax.spines["left"].set_color("#d4c2a8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.bar_label(bars, color="#29251f", padding=3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if cv_scores is not None and cv_predictions is not None:
        st.subheader("Cross-validation confusion matrix")
        cm = confusion_matrix(
            training_df["label"],
            cv_predictions,
            labels=LABELS,
        )
        fig2, ax2 = plt.subplots(figsize=(7, 4.8))
        fig2.patch.set_facecolor("#f3eadc")
        ax2.set_facecolor("#fffaf2")
        image = ax2.imshow(cm, cmap="bone")
        ax2.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
        ax2.set_yticks(range(len(LABELS)), LABELS)
        ax2.set_xlabel("Predicted label", color="#29251f")
        ax2.set_ylabel("Actual label", color="#29251f")
        ax2.set_title("Predicted vs actual labels", color="#29251f")
        ax2.tick_params(colors="#29251f")
        threshold = cm.max() / 2 if cm.size else 0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax2.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center",
                    color="#fffaf2" if cm[row, col] > threshold else "#29251f",
                )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Decision tree")
    fig3, ax3 = plt.subplots(figsize=(15, 7))
    fig3.patch.set_facecolor("#f3eadc")
    ax3.set_facecolor("#f3eadc")
    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        ax=ax3,
        node_ids=False,
        fontsize=8,
    )
    for patch in ax3.patches:
        patch.set_edgecolor("#29251f")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


with data_tab:
    st.subheader("Training data")
    st.dataframe(training_df, use_container_width=True, hide_index=True)
    st.subheader("Feature definitions")
    feature_info = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Meaning": [
            "Number of classes",
            "Total methods",
            "Total attributes",
            "A method name contains create or factory",
            "An attribute name contains instance or singleton",
            "A method name contains notify, update or subscribe",
        ],
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

