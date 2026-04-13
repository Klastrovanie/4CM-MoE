"""
4CM-MoE — Sigmoid vs Torus Router Comparison (Large Scale)
===========================================================
2000 sentences | 64 experts | M4 Max optimized

TF-IDF + Sigmoid vs TF-IDF + Torus
BERT   + Sigmoid vs BERT   + Torus

Prior Art: April 13, 2026
Based on: PhD Dissertation, 2011
ACM Digital Library: https://dl.acm.org/doi/book/10.5555/2231522
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

NUM_EXPERTS  = 64    # DeepSeek-V3 style 
TOP_K        = 4     # select 4 out of 64
NUM_TOPICS   = 4
STEPS        = 500
LOG_EVERY    = 50
AUX_ALPHA    = 0.01
LR           = 1e-3
SENTENCES_PER_TOPIC = 500  # total 2000


# ──────────────────────────────────────────────
# Base Sentences ( 20 per topic → using templates)
# ──────────────────────────────────────────────

BASE_TOPICS = {
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
        "How do I implement a linked list in Python?",
        "What is the difference between SQL and NoSQL?",
        "How do I use Docker containers?",
        "Explain the concept of recursion in programming.",
        "What is a closure in JavaScript?",
        "How do I optimize database queries?",
        "What is the difference between TCP and UDP?",
        "How do I implement authentication in a web app?",
        "What is a microservices architecture?",
        "How do I use regular expressions in Python?",
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
        "Explain the concept of a vector space.",
        "What is the gradient of a scalar field?",
        "How do I compute a Taylor series expansion?",
        "What is the Central Limit Theorem?",
        "Explain the concept of probability distributions.",
        "What is a Markov chain?",
        "How do I solve a quadratic equation?",
        "What is the Laplace transform used for?",
        "Explain the concept of linear independence.",
        "What is the dot product of two vectors?",
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
        "I am feeling overwhelmed with too many responsibilities.",
        "What are some ways to improve my mood?",
        "I feel like nobody understands me.",
        "How do I make new friends as an adult?",
        "I am struggling with self confidence issues.",
        "What should I do when I feel hopeless?",
        "How do I forgive someone who hurt me?",
        "I feel jealous of my colleagues success.",
        "What are some hobbies I can try to feel better?",
        "How do I stop overthinking everything?",
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
        "What is the difference between type 1 and type 2 diabetes?",
        "How does the liver process alcohol?",
        "What are the symptoms of depression?",
        "How do antidepressants work in the brain?",
        "What causes migraines and how to treat them?",
        "What is the recommended daily intake of vitamin D?",
        "How does sleep affect the immune system?",
        "What are the risk factors for stroke?",
        "How does insulin regulate blood sugar?",
        "What is the difference between CT scan and MRI?",
    ],
}

TOPIC_NAMES  = list(BASE_TOPICS.keys())
TOPIC_LABELS = {name: i for i, name in enumerate(TOPIC_NAMES)}

# augmentation using templates
TEMPLATES = [
    "{}",
    "Could you help me with this: {}",
    "I need to understand: {}",
    "Please explain: {}",
    "Can you tell me about: {}",
    "What is the best way to handle: {}",
    "I am trying to learn about: {}",
    "Help me figure out: {}",
    "I have a question about: {}",
    "I would like to know more about: {}",
    "Can you walk me through: {}",
    "I need some advice on: {}",
    "What do you think about: {}",
    "How should I approach: {}",
    "Give me guidance on: {}",
    "I am confused about: {}",
    "Can you clarify: {}",
    "I want to learn: {}",
    "What is the best approach for: {}",
    "I need help understanding: {}",
    "Could you break down: {}",
    "I am struggling with: {}",
    "What should I know about: {}",
    "Help me understand: {}",
    "I need a clear explanation of: {}",
]


def augment_sentences(base_sentences, target=500):
    """augmentation"""
    augmented = []
    while len(augmented) < target:
        for sent in base_sentences:
            for tmpl in TEMPLATES:
                augmented.append(tmpl.format(sent))
                if len(augmented) >= target:
                    break
            if len(augmented) >= target:
                break
    return augmented[:target]


# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────

class SigmoidRouter(nn.Module):
    """
    DeepSeek-V3 style sigmoid router (approximation)
    s_{i,t} = sigmoid(u_t^T e_i)
    """
    def __init__(self, d_model, num_experts=64, top_k=4):
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
    f(x,y) = [|x|^a1 + |y|^b1] * exp(-(|x|^c + |y|^d))
    Based on PhD Dissertation, 2011
    """
    def __init__(self, d_model, num_experts=64, top_k=4, scale=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.scale       = scale
        d                = d_model if d_model % 2 == 0 else d_model + 1
        self.d_half      = d // 2

        self.c  = nn.Parameter(torch.tensor(2.0))
        self.d  = nn.Parameter(torch.tensor(2.0))
        self.a1 = nn.Parameter(torch.tensor(4.0))
        self.b1 = nn.Parameter(torch.tensor(4.0))

        self.E_x  = nn.Parameter(torch.randn(self.d_half, num_experts) * 0.01)
        self.E_y  = nn.Parameter(torch.randn(self.d_half, num_experts) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def torus_f(self, x, y):
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

def run_training(X, Y, router_class, name, d_model):
    router = router_class(d_model, NUM_EXPERTS, TOP_K)
    model  = Model(router)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)

    log_steps, log_ce, log_acc, log_c = [], [], [], []

    for step in range(STEPS):
        # Mini-batch 
        idx_batch = torch.randperm(len(X))[:256]
        xb, yb    = X[idx_batch], Y[idx_batch]

        logits, aux, _ = model(xb)
        ce   = F.cross_entropy(logits, yb)
        loss = ce + AUX_ALPHA * aux

        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                logits_all, _, _ = model(X)
                acc = (logits_all.argmax(-1) == Y).float().mean().item()
            log_steps.append(step)
            log_ce.append(ce.item())
            log_acc.append(acc * 100)
            log_c.append(router.c.item() if hasattr(router, 'c') else None)
            print(f"    step {step:>4} | CE {ce.item():.4f} | Acc {acc:.2%} "
                  + (f"| c {router.c.item():.4f}" if hasattr(router, 'c') else ""))

    # Final eval
    model.eval()
    with torch.no_grad():
        logits_all, _, idx_all = model(X)
        facc = (logits_all.argmax(-1) == Y).float().mean().item()

    # Expert usage analysis for Routing Collapse detection
    expert_matrix = np.zeros((NUM_TOPICS, NUM_EXPERTS))
    expert_usage  = np.zeros(NUM_EXPERTS)
    for ti, tn in enumerate(TOPIC_NAMES):
        mask  = (Y == TOPIC_LABELS[tn])
        t_idx = idx_all[mask]
        for e in t_idx.flatten():
            expert_matrix[ti, e.item()] += 1
            expert_usage[e.item()] += 1

    # Expert usage entropy (higher = more uniform distribution)
    usage_prob = expert_usage / expert_usage.sum()
    entropy    = -np.sum(usage_prob * np.log(usage_prob + 1e-9))
    max_entropy = np.log(NUM_EXPERTS)
    entropy_ratio = entropy / max_entropy  

    steps_to_90 = next(
        (s for s, a in zip(log_steps, log_acc) if a >= 90.0), STEPS
    )
    steps_to_100 = next(
        (s for s, a in zip(log_steps, log_acc) if a >= 100.0), STEPS
    )

    fp = {}
    if hasattr(router, 'c'):
        fp = {
            "c":  router.c.item(),
            "d":  router.d.item(),
            "a1": router.a1.item(),
            "b1": router.b1.item(),
        }

    print(f"  ✓ {name}: Acc={facc:.2%} | Steps→90%={steps_to_90} "
          f"| Expert Entropy={entropy_ratio:.3f}")

    return {
        "name":          name,
        "steps":         log_steps,
        "ce":            log_ce,
        "acc":           log_acc,
        "c":             log_c,
        "em":            expert_matrix,
        "eu":            expert_usage,
        "entropy":       entropy_ratio,
        "facc":          facc,
        "s90":           steps_to_90,
        "s100":          steps_to_100,
        "fp":            fp,
    }


# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────

def get_tfidf(sentences):
    vec = TfidfVectorizer(max_features=512)
    arr = vec.fit_transform(sentences).toarray()
    return torch.tensor(arr, dtype=torch.float32)


def get_bert(sentences, device):
    from transformers import AutoTokenizer, AutoModel
    print("  Loading BERT...")
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    # batch processing (process all 2000 sentences in batches to avoid OOM)
    batch_size = 64
    all_hidden = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt",
                     padding=True, truncation=True,
                     max_length=64).to(device)
        with torch.no_grad():
            out = bert(**inputs)
        all_hidden.append(out.last_hidden_state[:, 0, :].cpu())
        if (i // batch_size) % 5 == 0:
            print(f"    {i+len(batch)}/{len(sentences)} sentences processed...")

    print("  BERT done!")
    return torch.cat(all_hidden, dim=0)


# ──────────────────────────────────────────────
# Figure 1
# ──────────────────────────────────────────────

def plot_figure1(results):
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
    hmaps = {
        "TF-IDF + Sigmoid": "Blues",
        "TF-IDF + Torus":   "Greens",
        "BERT + Sigmoid":   "Oranges",
        "BERT + Torus":     "Reds",
    }

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Figure 1: 4CM-MoE TorusRouter vs Sigmoid Router\n"
        f"2000 sentences | {NUM_EXPERTS} Experts | Top-{TOP_K} | "
        "Prior Art: April 13, 2026",
        fontsize=14, fontweight="bold", y=0.99
    )

    # (a) CE Loss
    ax1 = fig.add_subplot(3, 4, 1)
    for r in results:
        ax1.plot(r["steps"], r["ce"],
                 color=colors[r["name"]], marker=markers[r["name"]],
                 markersize=4, label=r["name"])
    ax1.set_title("(a) CE Loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # (b) Accuracy
    ax2 = fig.add_subplot(3, 4, 2)
    for r in results:
        ax2.plot(r["steps"], r["acc"],
                 color=colors[r["name"]], marker=markers[r["name"]],
                 markersize=4, label=r["name"])
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(90,  color="gray", linestyle=":",  alpha=0.3)
    ax2.set_title("(b) Accuracy (%)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Acc %")
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # (c) Steps to 90%
    ax3 = fig.add_subplot(3, 4, 3)
    names = [r["name"] for r in results]
    s90   = [r["s90"] for r in results]
    bars  = ax3.bar(range(len(names)), s90,
                    color=[colors[n] for n in names], alpha=0.8)
    ax3.set_title(f"(c) Steps to 90% Accuracy")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("Steps"); ax3.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, s90):
        ax3.text(bar.get_x()+bar.get_width()/2, v+2, str(v),
                 ha="center", fontsize=9, fontweight="bold")

    # (d) Expert Entropy (Routing Collapse)
    ax4 = fig.add_subplot(3, 4, 4)
    entropies = [r["entropy"] for r in results]
    bars2 = ax4.bar(range(len(names)), entropies,
                    color=[colors[n] for n in names], alpha=0.8)
    ax4.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect uniform=1.0")
    ax4.set_title("(d) Expert Usage Entropy\n(1.0=uniform, 0=collapse)")
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax4.set_ylabel("Entropy Ratio"); ax4.set_ylim(0, 1.2)
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars2, entropies):
        ax4.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                 ha="center", fontsize=9, fontweight="bold")

    # (e~h) Expert Heatmap (showing top 16 experts for clarity)
    show_experts = 16
    for ri, r in enumerate(results):
        ax = fig.add_subplot(3, 4, 5+ri)
        # top 16 Experts only
        top_experts = np.argsort(r["eu"])[::-1][:show_experts]
        em_top = r["em"][:, top_experts]
        im = ax.imshow(em_top, cmap=hmaps[r["name"]],
                       aspect="auto", vmin=0)
        ax.set_title(
            f"({'efgh'[ri]}) {r['name']}\n"
            f"Heatmap Top-{show_experts} Experts (Acc={r['facc']:.0%})\n"
            f"Entropy={r['entropy']:.3f}",
            fontsize=8
        )
        ax.set_xticks(range(show_experts))
        ax.set_xticklabels([f"E{e}" for e in top_experts],
                           rotation=90, fontsize=6)
        ax.set_yticks(range(NUM_TOPICS))
        ax.set_yticklabels(TOPIC_NAMES, fontsize=8)
        plt.colorbar(im, ax=ax)

    # (i) Torus param c
    ax9 = fig.add_subplot(3, 4, (9, 10))
    for r in results:
        if r["fp"]:
            ax9.plot(r["steps"], r["c"],
                     color=colors[r["name"]], marker=markers[r["name"]],
                     markersize=4, label=r["name"])
    ax9.axhline(2.0, color="gray", linestyle="--", alpha=0.5, label="init=2.0")
    ax9.set_title("(i) Torus param c over training (Torus only)")
    ax9.set_xlabel("Step"); ax9.set_ylabel("c value")
    ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

    # (j) Summary Table
    ax10 = fig.add_subplot(3, 4, (11, 12))
    ax10.axis("off")
    td = [["Model", "Acc", "Steps→90%", "Entropy", "c final", "d final"]]
    for r in results:
        c = f"{r['fp']['c']:.3f}" if r["fp"] else "N/A"
        d = f"{r['fp']['d']:.3f}" if r["fp"] else "N/A"
        td.append([
            r["name"],
            f"{r['facc']:.1%}",
            str(r["s90"]),
            f"{r['entropy']:.3f}",
            c, d,
        ])
    tbl = ax10.table(cellText=td[1:], colLabels=td[0],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.8)
    ax10.set_title("(j) Summary", fontsize=10)

    plt.tight_layout()

    # save the plot
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figure1_4cm_moe_2000.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure 1 saved → {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS (M4 Max)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # augmentation
    print("\n" + "="*60)
    print(f"  Step 1: Data Augmentation ({SENTENCES_PER_TOPIC} per topic)")
    print("="*60)

    sentences, labels = [], []
    for name, base in BASE_TOPICS.items():
        augmented = augment_sentences(base, SENTENCES_PER_TOPIC)
        sentences.extend(augmented)
        labels.extend([TOPIC_LABELS[name]] * len(augmented))
        print(f"  {name:<12} → {len(augmented)} sentences")

    Y = torch.tensor(labels)
    print(f"\n  Total: {len(sentences)} sentences")

    # TF-IDF
    print("\n" + "="*60)
    print("  Step 2: TF-IDF Embeddings")
    print("="*60)
    X_tfidf = get_tfidf(sentences)
    d_tfidf  = X_tfidf.shape[1]
    print(f"  Shape: {X_tfidf.shape}")

    # BERT
    print("\n" + "="*60)
    print("  Step 3: BERT Embeddings")
    print("="*60)
    X_bert = get_bert(sentences, device)
    d_bert  = X_bert.shape[1]
    print(f"  Shape: {X_bert.shape}")

    # Training
    print("\n" + "="*60)
    print(f"  Step 4: Training (4 experiments | {NUM_EXPERTS} experts)")
    print("="*60)

    results = []

    print("\n[1/4] TF-IDF + Sigmoid")
    results.append(run_training(X_tfidf, Y, SigmoidRouter,
                                "TF-IDF + Sigmoid", d_tfidf))

    print("\n[2/4] TF-IDF + Torus")
    results.append(run_training(X_tfidf, Y, TorusRouter,
                                "TF-IDF + Torus", d_tfidf))

    print("\n[3/4] BERT + Sigmoid")
    results.append(run_training(X_bert, Y, SigmoidRouter,
                                "BERT + Sigmoid", d_bert))

    print("\n[4/4] BERT + Torus")
    results.append(run_training(X_bert, Y, TorusRouter,
                                "BERT + Torus", d_bert))

    # Figure 1
    print("\n" + "="*60)
    print("  Step 5: Generating Figure 1")
    print("="*60)
    plot_figure1(results)


if __name__ == "__main__":
    main()