"""
4CM-MoE — Softmax vs Sigmoid vs Torus Router (2000 Classes)
============================================================
2000 classes | 20 sentences per class | 40,000 total sentences
64 experts | M4 Max optimized

Class = (topic, template) combination
4 topics × 500 templates × 20 base sentences = 2000 classes

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

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_EXPERTS          = 64     # DeepSeek-V3 style
TOP_K                = 4      # select 4 out of 64
NUM_CLASSES          = 2000   # 4 topics × 500 templates × 20 base sentences
SENTENCES_PER_CLASS  = 20     # 20 base sentences per class
STEPS                = 650
LOG_EVERY            = 25
AUX_ALPHA            = 0.01   # load balancing loss weight
LR                   = 1e-3
BATCH_SIZE           = 512    # larger batch for 40k sentences


# ──────────────────────────────────────────────
# Base Sentences (20 per topic)
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

TOPIC_NAMES = list(BASE_TOPICS.keys())  # 4 topics

# 500 templates × 4 topics × 20 sentences = 2000 classes
_BASE_TEMPLATES = [
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
    "Can you give me an overview of: {}",
    "What are the key points about: {}",
    "I am curious about: {}",
    "Can you elaborate on: {}",
    "What is the main idea behind: {}",
    "I would appreciate your help with: {}",
    "Can you simplify: {}",
    "What are the basics of: {}",
    "I am interested in learning: {}",
    "Could you give me an example of: {}",
    "What are common mistakes in: {}",
    "How do experts think about: {}",
    "What is the most important thing about: {}",
    "Can you summarize: {}",
    "I want to get better at: {}",
    "What should a beginner know about: {}",
    "How do I get started with: {}",
    "What are the best practices for: {}",
    "Can you recommend resources for: {}",
    "What are the pros and cons of: {}",
    "How do professionals handle: {}",
    "What is the history of: {}",
    "Can you compare different approaches to: {}",
    "What are the latest developments in: {}",
    "How has the field evolved regarding: {}",
]

# generate 500 templates by adding prefixes/suffixes to base templates
_PREFIXES = [
    "", "Quick question: ", "Just wondering: ", "Out of curiosity: ",
    "For my project: ", "For my research: ", "For a friend: ",
    "In simple terms: ", "In detail: ", "Step by step: ",
    "As a beginner: ", "As an expert: ",
]

_SUFFIXES = [
    "", " Please be detailed.", " Keep it simple.",
    " Give me an example.", " Use simple language.",
    " Be concise.", " Explain like I am five.",
    " Give me a practical example.", " Focus on the key points.",
    " Include common pitfalls.",
]

TEMPLATES = []
for prefix in _PREFIXES:
    for suffix in _SUFFIXES:
        for tmpl in _BASE_TEMPLATES[:5]:  # use first 5 base templates
            t = prefix + tmpl + suffix
            TEMPLATES.append(t)
            if len(TEMPLATES) >= 500:
                break
        if len(TEMPLATES) >= 500:
            break
    if len(TEMPLATES) >= 500:
        break

TEMPLATES = TEMPLATES[:500]
assert len(TEMPLATES) == 500, f"Need exactly 500 templates, got {len(TEMPLATES)}"


def build_dataset():
    """
    Build 2000-class dataset.
    Class = (topic_idx, template_idx)
    Each class has 20 sentences (base sentences with that template applied).
    Total: 4 × 500 × 20 = 2000 classes × 20 sentences = 40,000 sentences
    """
    sentences = []
    labels    = []
    class_id  = 0

    class_info = []  # for analysis: (topic, template)

    for topic_name, base_sents in BASE_TOPICS.items():
        for tmpl_idx, tmpl in enumerate(TEMPLATES):
            # apply this template to all 20 base sentences → 20 sentences for this class
            for sent in base_sents:
                sentences.append(tmpl.format(sent))
                labels.append(class_id)
            class_info.append((topic_name, tmpl_idx))
            class_id += 1

    assert class_id == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {class_id}"
    assert len(sentences) == NUM_CLASSES * SENTENCES_PER_CLASS

    return sentences, labels, class_info


# ──────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────

class SoftmaxRouter(nn.Module):
    """
    Classic Softmax Router — Switch Transformer, Mixtral
    sum=1 → expert competition → Routing Collapse worsens with more classes
    """
    def __init__(self, d_model, num_experts=64, top_k=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.E    = nn.Parameter(torch.randn(d_model, num_experts) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, u):
        scores          = F.softmax(u @ self.E + self.bias, dim=-1)
        topk_s, topk_i = torch.topk(scores, self.top_k, dim=-1)
        aux_loss        = (scores.mean(0) ** 2).sum() * self.num_experts
        return topk_i, topk_s, scores, aux_loss


class SigmoidRouter(nn.Module):
    """
    DeepSeek-V3 style sigmoid router (approximation)
    Independent scoring → no expert competition
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
    4-peak structure → structural Routing Collapse prevention
    Based on PhD Dissertation, 2011
    """
    def __init__(self, d_model, num_experts=64, top_k=4, scale=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.scale       = scale
        d                = d_model if d_model % 2 == 0 else d_model + 1
        self.d_half      = d // 2

        # learnable torus shape parameters
        self.c  = nn.Parameter(torch.tensor(2.0))
        self.d  = nn.Parameter(torch.tensor(2.0))
        self.a1 = nn.Parameter(torch.tensor(4.0))
        self.b1 = nn.Parameter(torch.tensor(4.0))

        # expert centroid vectors — x-axis and y-axis
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
    """Router scores → linear head → 2000-class prediction."""
    def __init__(self, router, num_classes=2000):
        super().__init__()
        self.router = router
        self.head   = nn.Linear(NUM_EXPERTS, num_classes)

    def forward(self, u):
        idx, s, all_s, aux = self.router(u)
        return self.head(all_s), aux, idx


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def run_training(X, Y, router_class, name, d_model):
    """Train a single router on 2000-class task."""
    router = router_class(d_model, NUM_EXPERTS, TOP_K)
    model  = Model(router, num_classes=NUM_CLASSES)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)

    log_steps, log_ce, log_acc, log_c = [], [], [], []

    for step in range(STEPS):
        # random mini-batch
        idx_batch = torch.randperm(len(X))[:BATCH_SIZE]
        xb, yb    = X[idx_batch], Y[idx_batch]

        logits, aux, _ = model(xb)
        ce   = F.cross_entropy(logits, yb)
        loss = ce + AUX_ALPHA * aux

        opt.zero_grad(); loss.backward(); opt.step()

        if step % LOG_EVERY == 0:
            with torch.no_grad():
                # evaluate on subset (40k is large)
                eval_idx = torch.randperm(len(X))[:2000]
                logits_eval, _, _ = model(X[eval_idx])
                acc = (logits_eval.argmax(-1) == Y[eval_idx]).float().mean().item()
            log_steps.append(step)
            log_ce.append(ce.item())
            log_acc.append(acc * 100)
            log_c.append(router.c.item() if hasattr(router, 'c') else None)
            print(f"    step {step:>4} | CE {ce.item():.4f} | Acc {acc:.2%} "
                  + (f"| c {router.c.item():.4f}" if hasattr(router, 'c') else ""))

    # final evaluation
    model.eval()
    with torch.no_grad():
        eval_idx  = torch.randperm(len(X))[:5000]
        logits_f, _, idx_all = model(X[eval_idx])
        facc = (logits_f.argmax(-1) == Y[eval_idx]).float().mean().item()

    # expert usage entropy
    expert_usage = np.zeros(NUM_EXPERTS)
    for e in idx_all.flatten():
        expert_usage[e.item()] += 1
    usage_prob    = expert_usage / expert_usage.sum()
    entropy       = -np.sum(usage_prob * np.log(usage_prob + 1e-9))
    entropy_ratio = entropy / np.log(NUM_EXPERTS)

    steps_to_10 = next(
        (s for s, a in zip(log_steps, log_acc) if a >= 10.0), STEPS
    )
    steps_to_50 = next(
        (s for s, a in zip(log_steps, log_acc) if a >= 50.0), STEPS
    )

    fp = {}
    if hasattr(router, 'c'):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{name.replace(' ', '_').replace('+', '')}_router.pt"
        )
        torch.save(router.state_dict(), save_path)
        print(f"  → Router saved: {save_path}")

        fp = {
            "c":  router.c.item(),
            "d":  router.d.item(),
            "a1": router.a1.item(),
            "b1": router.b1.item(),
        }

    print(f"  ✓ {name}: Acc={facc:.2%} | Steps→10%={steps_to_10} "
          f"| Steps→50%={steps_to_50} | Entropy={entropy_ratio:.3f}")

    return {
        "name":    name,
        "steps":   log_steps,
        "ce":      log_ce,
        "acc":     log_acc,
        "c":       log_c,
        "eu":      expert_usage,
        "entropy": entropy_ratio,
        "facc":    facc,
        "s10":     steps_to_10,
        "s50":     steps_to_50,
        "fp":      fp,
    }

def plot_scatter(router, X_bert, Y, name, out_dir):
    """x-y routing space scatter plot by topic (TorusRouter only)."""
    if not hasattr(router, 'E_x'):
        return

    topic_colors = ['#2196F3', '#55A868', '#FF9800', '#C44E52']
    
    fig, axes = plt.subplots(8, 8, figsize=(32, 32))
    fig.suptitle(
        f"Figure 2: {name} — x-y Routing Space by Topic\n"
        "Each point = one sentence | Color = topic",
        fontsize=12, fontweight="bold"
    )

    router.eval()
    with torch.no_grad():
        ux = X_bert[:, :router.d_half]
        uy = X_bert[:, router.d_half:router.d_half*2]

        for ei, ax in enumerate(axes.flatten()):
            x = torch.tanh(ux @ router.E_x[:, ei]) * router.scale
            y = torch.tanh(uy @ router.E_y[:, ei]) * router.scale

            for ti, topic in enumerate(TOPIC_NAMES):
                # select 500 classes per topic
                mask = (Y >= ti * 500) & (Y < (ti + 1) * 500)
                # then sample 100 out of them
                idx_sample = torch.where(mask)[0][:100]
                ax.scatter(
                    x[idx_sample].numpy(),
                    y[idx_sample].numpy(),
                    color=topic_colors[ti],
                    alpha=0.4, s=3,
                    label=topic if ei == 0 else ""
                )

            ax.set_title(f"E{ei}", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.axhline(0, color='gray', alpha=0.3)
            ax.axvline(0, color='gray', alpha=0.3)
            ax.grid(alpha=0.2)

    # legend
    handles = [plt.scatter([], [], color=c, label=t)
               for t, c in zip(TOPIC_NAMES, topic_colors)]
    fig.legend(handles=handles, loc='lower center',
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(out_dir, f"{name.replace(' ', '_').replace('+', '')}_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Scatter plot saved: {out}")


# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────

def get_tfidf(sentences):
    """TF-IDF embeddings — no external model required."""
    vec = TfidfVectorizer(max_features=512)
    arr = vec.fit_transform(sentences).toarray()
    return torch.tensor(arr, dtype=torch.float32)


def get_bert(sentences, device):
    """BERT CLS token embeddings — batch processing for 40k sentences."""
    from transformers import AutoTokenizer, AutoModel
    print("  Loading BERT...")
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    batch_size = 128  # larger batch for MPS
    all_hidden = []
    for i in range(0, len(sentences), batch_size):
        batch  = sentences[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt",
                     padding=True, truncation=True,
                     max_length=64).to(device)
        with torch.no_grad():
            out = bert(**inputs)
        all_hidden.append(out.last_hidden_state[:, 0, :].cpu())
        if i % 5000 == 0:
            print(f"    {i+len(batch)}/{len(sentences)} processed...")

    print("  BERT done!")
    return torch.cat(all_hidden, dim=0)


# ──────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────

def plot_figure(results):
    """Generate comparison figure for 2000-class experiment."""
    colors = {
        "TF-IDF + Softmax": "#2196F3",
        "TF-IDF + Sigmoid": "#4C72B0",
        "TF-IDF + Torus":   "#55A868",
        "BERT + Softmax":   "#FF9800",
        "BERT + Sigmoid":   "#DD8452",
        "BERT + Torus":     "#C44E52",
    }
    markers = {
        "TF-IDF + Softmax": "v",
        "TF-IDF + Sigmoid": "o",
        "TF-IDF + Torus":   "s",
        "BERT + Softmax":   "P",
        "BERT + Sigmoid":   "^",
        "BERT + Torus":     "D",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Figure 1: 4CM-MoE — 2000-Class Experiment\n"
        f"40,000 sentences | {NUM_EXPERTS} Experts | Top-{TOP_K} | "
        "Softmax Routing Collapse at Scale | Prior Art: April 13, 2026",
        fontsize=13, fontweight="bold"
    )

    # (a) CE Loss
    ax = axes[0, 0]
    for r in results:
        ax.plot(r["steps"], r["ce"],
                color=colors[r["name"]], marker=markers[r["name"]],
                markersize=3, label=r["name"])
    ax.set_title("(a) CE Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) Accuracy
    ax = axes[0, 1]
    for r in results:
        ax.plot(r["steps"], r["acc"],
                color=colors[r["name"]], marker=markers[r["name"]],
                markersize=3, label=r["name"])
    ax.set_title("(b) Accuracy (%) — 2000 classes")
    ax.set_xlabel("Step"); ax.set_ylabel("Acc %")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) Steps to 10%
    ax = axes[0, 2]
    names = [r["name"] for r in results]
    s10   = [r["s10"] for r in results]
    bars  = ax.bar(range(len(names)), s10,
                   color=[colors[n] for n in names], alpha=0.8)
    ax.set_title("(c) Steps to 10% Accuracy")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Steps"); ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, s10):
        ax.text(bar.get_x()+bar.get_width()/2, v+1, str(v),
                ha="center", fontsize=8, fontweight="bold")

    # (d) Expert Usage Entropy
    ax = axes[1, 0]
    entropies = [r["entropy"] for r in results]
    bars2 = ax.bar(range(len(names)), entropies,
                   color=[colors[n] for n in names], alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect=1.0")
    ax.set_title("(d) Expert Usage Entropy\n(1.0=uniform, 0=collapse)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Entropy Ratio"); ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars2, entropies):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=8, fontweight="bold")

    # (e) Torus param c
    ax = axes[1, 1]
    for r in results:
        if r["fp"]:
            ax.plot(r["steps"], r["c"],
                    color=colors[r["name"]], marker=markers[r["name"]],
                    markersize=3, label=r["name"])
    ax.axhline(2.0, color="gray", linestyle="--", alpha=0.5, label="init=2.0")
    ax.set_title("(e) Torus param c (Torus only)")
    ax.set_xlabel("Step"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (f) Summary Table
    ax = axes[1, 2]
    ax.axis("off")
    td = [["Model", "Acc", "→10%", "→50%", "Entropy", "c"]]
    for r in results:
        c = f"{r['fp']['c']:.3f}" if r["fp"] else "N/A"
        td.append([
            r["name"],
            f"{r['facc']:.1%}",
            str(r["s10"]),
            str(r["s50"]),
            f"{r['entropy']:.3f}",
            c,
        ])
    tbl = ax.table(cellText=td[1:], colLabels=td[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.1, 1.6)
    ax.set_title("(f) Summary — 2000 Classes", fontsize=10)

    plt.tight_layout()

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "figure1_4cm_moe_2000class.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure 1 saved → {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # device selection + seed fixing
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(SEED)
        print("Device: Apple MPS (M4 Max)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # step 1: build 2000-class dataset
    print("\n" + "="*60)
    print("  Step 1: Building 2000-Class Dataset")
    print("="*60)
    sentences, labels, class_info = build_dataset()
    Y = torch.tensor(labels)
    print(f"  Classes:   {NUM_CLASSES}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Labels:    {Y.shape}")

    # step 2: TF-IDF embeddings
    print("\n" + "="*60)
    print("  Step 2: TF-IDF Embeddings")
    print("="*60)
    X_tfidf = get_tfidf(sentences)
    d_tfidf  = X_tfidf.shape[1]
    print(f"  Shape: {X_tfidf.shape}")

    # step 3: BERT embeddings
    print("\n" + "="*60)
    print("  Step 3: BERT Embeddings (40,000 sentences)")
    print("="*60)
    X_bert = get_bert(sentences, device)
    d_bert  = X_bert.shape[1]
    print(f"  Shape: {X_bert.shape}")

    # step 4: training — 6 experiments
    print("\n" + "="*60)
    print(f"  Step 4: Training (6 experiments | {NUM_CLASSES} classes | {NUM_EXPERTS} experts)")
    print("="*60)

    results = []

    print("\n[1/6] TF-IDF + Softmax")
    results.append(run_training(X_tfidf, Y, SoftmaxRouter,
                                "TF-IDF + Softmax", d_tfidf))

    print("\n[2/6] TF-IDF + Sigmoid")
    results.append(run_training(X_tfidf, Y, SigmoidRouter,
                                "TF-IDF + Sigmoid", d_tfidf))

    print("\n[3/6] TF-IDF + Torus")
    results.append(run_training(X_tfidf, Y, TorusRouter,
                                "TF-IDF + Torus", d_tfidf))

    print("\n[4/6] BERT + Softmax")
    results.append(run_training(X_bert, Y, SoftmaxRouter,
                                "BERT + Softmax", d_bert))

    print("\n[5/6] BERT + Sigmoid")
    results.append(run_training(X_bert, Y, SigmoidRouter,
                                "BERT + Sigmoid", d_bert))

    print("\n[6/6] BERT + Torus")
    results.append(run_training(X_bert, Y, TorusRouter,
                                "BERT + Torus", d_bert))

    # step 5: generate figure
    print("\n" + "="*60)
    print("  Step 5: Generating Figure 1")
    print("="*60)
    plot_figure(results)

    # step 6: scatter plot (TorusRouter only)
    print("\n" + "="*60)
    print("  Step 6: Generating Scatter Plots")
    print("="*60)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for r in results:
        if "Torus" in r["name"] and "BERT" in r["name"]:
            # load the router without training then plot scatter (to avoid GPU memory issues) 
            router = TorusRouter(d_bert, NUM_EXPERTS, TOP_K)
            pt_path = os.path.join(
                out_dir,
                f"{r['name'].replace(' ', '_').replace('+', '')}_router.pt"
            )
            if os.path.exists(pt_path):
                router.load_state_dict(torch.load(pt_path, weights_only=True))
                plot_scatter(router, X_bert, Y, r["name"], out_dir)


if __name__ == "__main__":
    main()
