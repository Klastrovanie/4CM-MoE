"""
4CM-MoE — Sigmoid vs Torus Router Comparison
=============================================
TF-IDF + Sigmoid vs TF-IDF + Torus
BERT   + Sigmoid vs BERT   + Torus

Generates Figure 1 automatically after training.

Prior Art: April 13, 2026
Based on: PhD Dissertation, 2011
ACM Digital Library: https://dl.acm.org/doi/book/10.5555/2231522
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────

TOPICS = {
    "coding": [
        "Write a Python function to sort a list.",
        "Debug this JavaScript code for me.",
        "How do I use async await in Python?",
        "Explain what a REST API is.",
        "What is the difference between git merge and rebase?",
        "How do I connect to a PostgreSQL database?",
        "What does the map function do in JavaScript?",
        "How do I handle exceptions in Python?",
        "Explain object oriented programming concepts.",
        "What is the time complexity of binary search?",
    ],
    "math": [
        "Solve this differential equation for me.",
        "Calculate the eigenvalues of this matrix.",
        "What is the derivative of sin x function?",
        "Explain the Pythagorean theorem with example.",
        "How do I integrate x squared dx?",
        "What is a Fourier transform used for?",
        "Explain Bayes theorem with a practical example.",
        "What is the determinant of a 3x3 matrix?",
        "How do I solve a system of linear equations?",
        "What is the limit of sin x over x as x approaches zero?",
    ],
    "emotion": [
        "I feel sad today what should I do?",
        "Can you recommend a good movie to watch?",
        "How do I make pasta carbonara at home?",
        "I am feeling anxious about my job interview tomorrow.",
        "What are some tips for better sleep quality?",
        "I had a fight with my best friend yesterday.",
        "How do I stay motivated when I feel like giving up?",
        "What should I do on a rainy day at home?",
        "I feel lonely living alone in a new city.",
        "How do I deal with stress and anxiety at work?",
    ],
    "medical": [
        "What are the symptoms of diabetes disease?",
        "Explain how vaccines work in the human body.",
        "What is the recommended dosage for ibuprofen?",
        "What are the early signs of a heart attack?",
        "How does the immune system fight bacterial infections?",
        "What is the difference between a virus and bacteria?",
        "What causes high blood pressure in adults?",
        "How does chemotherapy treatment work for cancer?",
        "What are the common side effects of antibiotics?",
        "Explain what an MRI scan is used for.",
    ],
}

TOPIC_NAMES  = list(TOPICS.keys())
TOPIC_LABELS = {name: i for i, name in enumerate(TOPIC_NAMES)}
NUM_TOPICS   = len(TOPIC_NAMES)
NUM_EXPERTS  = 8
TOP_K        = 2
STEPS        = 300
LOG_EVERY    = 30


# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────

class SigmoidRouter(nn.Module):
    """
    DeepSeek-V3 style sigmoid router (approximation)
    s_{i,t} = sigmoid(u_t^T e_i)
    e_i: learned centroid (approximated — actual DeepSeek weights not public)
    """
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.E    = nn.Parameter(torch.randn(d_model, num_experts) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, u):
        scores          = torch.sigmoid(u @ self.E + self.bias)
        topk_s, topk_i = torch.topk(scores, self.top_k, dim=-1)
        probs           = F.softmax(scores, dim=-1)
        aux_loss        = (probs.mean(0) ** 2).sum() * self.num_experts
        return topk_i, topk_s, scores, aux_loss


class TorusRouter(nn.Module):
    """
    4CM Torus Router
    s_{i,t} = f(u_t^T e_i^x, u_t^T e_i^y)
    f(x,y) = [|x|^a1 + |y|^b1] * exp(-(|x|^c + |y|^d))
    Based on PhD Dissertation, 2011
    """
    def __init__(self, d_model, num_experts=8, top_k=2, scale=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.scale       = scale
        d                = d_model if d_model % 2 == 0 else d_model + 1
        self.d_half      = d // 2

        # Learnable torus shape parameters
        self.c  = nn.Parameter(torch.tensor(2.0))
        self.d  = nn.Parameter(torch.tensor(2.0))
        self.a1 = nn.Parameter(torch.tensor(4.0))
        self.b1 = nn.Parameter(torch.tensor(4.0))

        # Expert centroid vectors — x axis, y axis
        self.E_x  = nn.Parameter(torch.randn(self.d_half, num_experts) * 0.01)
        self.E_y  = nn.Parameter(torch.randn(self.d_half, num_experts) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def torus_f(self, x, y):
        """f(x,y) = [|x|^a1 + |y|^b1] * exp(-(|x|^c + |y|^d))"""
        xa, ya = torch.abs(x), torch.abs(y)
        return (xa ** self.a1 + ya ** self.b1) * \
               torch.exp(-(xa ** self.c + ya ** self.d))

    def forward(self, u):
        if u.shape[-1] % 2 != 0:
            u = F.pad(u, (0, 1))
        ux = u[..., :self.d_half]
        uy = u[..., self.d_half:]
        x       = torch.tanh(ux @ self.E_x) * self.scale
        y       = torch.tanh(uy @ self.E_y) * self.scale
        scores  = self.torus_f(x, y) + self.bias
        topk_s, topk_i = torch.topk(scores, self.top_k, dim=-1)
        probs    = F.softmax(scores, dim=-1)
        aux_loss = (probs.mean(0) ** 2).sum() * self.num_experts
        return topk_i, topk_s, scores, aux_loss


class Model(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router
        self.head   = nn.Linear(NUM_EXPERTS, NUM_TOPICS)

    def forward(self, u):
        idx, s, all_s, aux = self.router(u)
        return self.head(all_s), aux, idx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def run_training(X, Y, router_class, router_name, d_model):
    router = router_class(d_model, NUM_EXPERTS, TOP_K)
    model  = Model(router)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_steps, log_ce, log_acc = [], [], []
    log_c, log_d = [], []

    for step in range(STEPS):
        logits, aux, idx = model(X)
        ce   = F.cross_entropy(logits, Y)
        loss = ce + 0.01 * aux
        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0:
            acc = (logits.argmax(-1) == Y).float().mean().item()
            log_steps.append(step)
            log_ce.append(ce.item())
            log_acc.append(acc * 100)
            if hasattr(model.router, 'c'):
                log_c.append(model.router.c.item())
                log_d.append(model.router.d.item())
            else:
                log_c.append(None)
                log_d.append(None)

    # Expert selection pattern
    model.eval()
    with torch.no_grad():
        logits, _, idx = model(X)
        final_acc = (logits.argmax(-1) == Y).float().mean().item()

    expert_matrix = np.zeros((NUM_TOPICS, NUM_EXPERTS))
    for ti, topic in enumerate(TOPIC_NAMES):
        lbl   = TOPIC_LABELS[topic]
        mask  = (Y == lbl)
        t_idx = idx[mask]
        for e in t_idx.flatten():
            expert_matrix[ti, e.item()] += 1

    # Steps to 100%
    steps_to_100 = next(
        (s for s, a in zip(log_steps, log_acc) if a >= 100.0),
        STEPS
    )

    # Final torus params
    final_params = {}
    if hasattr(model.router, 'c'):
        final_params = {
            "c":  model.router.c.item(),
            "d":  model.router.d.item(),
            "a1": model.router.a1.item(),
            "b1": model.router.b1.item(),
        }

    print(f"  {router_name}: Acc={final_acc:.0%} | Steps to 100%={steps_to_100}")

    return {
        "name":          router_name,
        "steps":         log_steps,
        "ce":            log_ce,
        "acc":           log_acc,
        "c":             log_c,
        "d":             log_d,
        "expert_matrix": expert_matrix,
        "final_acc":     final_acc,
        "steps_to_100":  steps_to_100,
        "final_params":  final_params,
    }


# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────

def get_tfidf(sentences):
    vec = TfidfVectorizer(max_features=256)
    arr = vec.fit_transform(sentences).toarray()
    return torch.tensor(arr, dtype=torch.float32)


def get_bert(sentences, device):
    from transformers import AutoTokenizer, AutoModel
    print("  Loading BERT...")
    tok   = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert  = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()
    inputs = tok(sentences, return_tensors="pt",
                 padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = bert(**inputs)
    print("  BERT loaded!")
    return out.last_hidden_state[:, 0, :].cpu()


# ──────────────────────────────────────────────
# Figure 1
# ──────────────────────────────────────────────

def plot_figure1(results):
    """
    2x2 experiments:
    TF-IDF + Sigmoid | TF-IDF + Torus
    BERT   + Sigmoid | BERT   + Torus
    """
    colors = {
        "TF-IDF + Sigmoid": "#4C72B0",
        "TF-IDF + Torus":   "#55A868",
        "BERT + Sigmoid":   "#DD8452",
        "BERT + Torus":     "#C44E52",
    }
    markers = {
        "TF-IDF + Sigmoid": "o",
        "TF-IDF + Torus":   "s",
        "BERT + Sigmoid":   "^",
        "BERT + Torus":     "D",
    }

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Figure 1: 4CM-MoE TorusRouter vs Sigmoid Router\n"
        "TF-IDF vs BERT Embeddings | Prior Art: April 13, 2026",
        fontsize=14, fontweight="bold", y=0.99
    )

    # ── (a) CE Loss ──
    ax1 = fig.add_subplot(3, 4, 1)
    for r in results:
        ax1.plot(r["steps"], r["ce"],
                 color=colors[r["name"]],
                 marker=markers[r["name"]],
                 markersize=4, label=r["name"])
    ax1.set_title("(a) CE Loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # ── (b) Accuracy ──
    ax2 = fig.add_subplot(3, 4, 2)
    for r in results:
        ax2.plot(r["steps"], r["acc"],
                 color=colors[r["name"]],
                 marker=markers[r["name"]],
                 markersize=4, label=r["name"])
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("(b) Accuracy (%)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # ── (c) Steps to 100% ──
    ax3 = fig.add_subplot(3, 4, 3)
    names = [r["name"] for r in results]
    s100  = [r["steps_to_100"] for r in results]
    bars  = ax3.bar(range(len(names)), s100,
                    color=[colors[n] for n in names], alpha=0.8)
    ax3.set_title("(c) Steps to 100% Accuracy")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Steps")
    ax3.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, s100):
        ax3.text(bar.get_x() + bar.get_width()/2, v+2, str(v),
                ha="center", fontsize=9, fontweight="bold")

    # ── (d) Torus c param (Torus only) ──
    ax4 = fig.add_subplot(3, 4, 4)
    for r in results:
        if r["final_params"]:
            ax4.plot(r["steps"], r["c"],
                     color=colors[r["name"]],
                     marker=markers[r["name"]],
                     markersize=4, label=r["name"])
    ax4.axhline(2.0, color="gray", linestyle="--", alpha=0.5, label="init=2.0")
    ax4.set_title("(d) Torus param c (Torus only)")
    ax4.set_xlabel("Step"); ax4.set_ylabel("c value")
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    # ── (e~h) Expert Heatmaps ──
    hmap_colors = {
        "TF-IDF + Sigmoid": "Blues",
        "TF-IDF + Torus":   "Greens",
        "BERT + Sigmoid":   "Oranges",
        "BERT + Torus":     "Reds",
    }
    for ri, r in enumerate(results):
        ax = fig.add_subplot(3, 4, 5 + ri)
        im = ax.imshow(r["expert_matrix"],
                       cmap=hmap_colors[r["name"]],
                       aspect="auto", vmin=0, vmax=10)
        ax.set_title(f"({'efgh'[ri]}) {r['name']}\nExpert Heatmap (Acc={r['final_acc']:.0%})",
                     fontsize=9)
        ax.set_xticks(range(NUM_EXPERTS))
        ax.set_xticklabels([f"E{i}" for i in range(NUM_EXPERTS)], fontsize=8)
        ax.set_yticks(range(NUM_TOPICS))
        ax.set_yticklabels(TOPIC_NAMES, fontsize=8)
        plt.colorbar(im, ax=ax)
        for i in range(NUM_TOPICS):
            for j in range(NUM_EXPERTS):
                v = r["expert_matrix"][i, j]
                if v > 0:
                    ax.text(j, i, int(v), ha="center", va="center",
                            fontsize=8,
                            color="white" if v > 6 else "black")

    # ── (i) Final Torus Params ──
    ax9 = fig.add_subplot(3, 4, (9, 10))
    torus_results = [r for r in results if r["final_params"]]
    param_names   = ["c", "d", "a1", "b1"]
    init_vals     = [2.0, 2.0, 4.0, 4.0]
    x = np.arange(len(param_names))
    w = 0.2

    ax9.bar(x - w, init_vals, w, label="Init", color="gray", alpha=0.5)
    for ri, r in enumerate(torus_results):
        vals = [r["final_params"]["c"],  r["final_params"]["d"],
                r["final_params"]["a1"], r["final_params"]["b1"]]
        ax9.bar(x + ri*w, vals, w,
                label=r["name"],
                color=colors[r["name"]], alpha=0.8)
        for i, (v, iv) in enumerate(zip(vals, init_vals)):
            arrow = "↑" if v > iv else "↓"
            ax9.text(i + ri*w, v + 0.05,
                     f"{arrow}{abs(v-iv):.2f}",
                     ha="center", fontsize=7,
                     color=colors[r["name"]])

    ax9.set_title("(i) Final Torus Params: Init vs Learned")
    ax9.set_xticks(x + w/2)
    ax9.set_xticklabels(["c (decay x)", "d (decay y)",
                          "a1 (poly x)", "b1 (poly y)"])
    ax9.legend(fontsize=8); ax9.grid(alpha=0.3, axis="y")

    # ── (j) Summary Table ──
    ax10 = fig.add_subplot(3, 4, (11, 12))
    ax10.axis("off")
    table_data = [["Model", "Acc", "Steps→100%", "c final", "d final"]]
    for r in results:
        c_val = f"{r['final_params']['c']:.3f}" if r["final_params"] else "N/A"
        d_val = f"{r['final_params']['d']:.3f}" if r["final_params"] else "N/A"
        table_data.append([
            r["name"],
            f"{r['final_acc']:.0%}",
            str(r["steps_to_100"]),
            c_val,
            d_val,
        ])
    tbl = ax10.table(cellText=table_data[1:],
                     colLabels=table_data[0],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    ax10.set_title("(j) Summary", fontsize=10)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure1_4cm_moe_full.png")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure 1 saved → {out}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Sentences & Labels
    sentences, labels = [], []
    for name, ss in TOPICS.items():
        sentences.extend(ss)
        labels.extend([TOPIC_LABELS[name]] * len(ss))
    Y = torch.tensor(labels)

    print("\n" + "="*60)
    print("  Step 1: TF-IDF Embeddings")
    print("="*60)
    X_tfidf = get_tfidf(sentences)
    d_tfidf  = X_tfidf.shape[1]
    print(f"  Shape: {X_tfidf.shape}")

    print("\n" + "="*60)
    print("  Step 2: BERT Embeddings")
    print("="*60)
    X_bert = get_bert(sentences, device)
    d_bert  = X_bert.shape[1]
    print(f"  Shape: {X_bert.shape}")

    print("\n" + "="*60)
    print("  Step 3: Training (4 experiments)")
    print("="*60)

    results = []

    print("\n[1/4] TF-IDF + Sigmoid")
    results.append(run_training(X_tfidf, Y, SigmoidRouter,
                                "TF-IDF + Sigmoid", d_tfidf))

    print("[2/4] TF-IDF + Torus")
    results.append(run_training(X_tfidf, Y, TorusRouter,
                                "TF-IDF + Torus", d_tfidf))

    print("[3/4] BERT + Sigmoid")
    results.append(run_training(X_bert, Y, SigmoidRouter,
                                "BERT + Sigmoid", d_bert))

    print("[4/4] BERT + Torus")
    results.append(run_training(X_bert, Y, TorusRouter,
                                "BERT + Torus", d_bert))

    print("\n" + "="*60)
    print("  Step 4: Generating Figure 1")
    print("="*60)
    plot_figure1(results)


if __name__ == "__main__":
    main()
