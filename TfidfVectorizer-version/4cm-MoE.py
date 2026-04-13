import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer

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

class TorusRouter(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2, scale=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k  = top_k
        self.scale  = scale
        self.c  = nn.Parameter(torch.tensor(2.0))
        self.d  = nn.Parameter(torch.tensor(2.0))
        self.a1 = nn.Parameter(torch.tensor(4.0))
        self.b1 = nn.Parameter(torch.tensor(4.0))
        # if d_model -> odd, make it even, even steven.
        d = d_model if d_model % 2 == 0 else d_model + 1
        self.E_x  = nn.Parameter(torch.randn(d//2, num_experts)*0.01)
        self.E_y  = nn.Parameter(torch.randn(d//2, num_experts)*0.01)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.d_half = d // 2

    def torus_f(self, x, y):
        xa, ya = torch.abs(x), torch.abs(y)
        return (xa**self.a1 + ya**self.b1) * torch.exp(-(xa**self.c + ya**self.d))

    def forward(self, u):
        # if d_model -> odd, pad with zero 
        if u.shape[-1] % 2 != 0:
            u = F.pad(u, (0, 1))
        ux = u[..., :self.d_half]
        uy = u[..., self.d_half:]
        x = torch.tanh(ux @ self.E_x) * self.scale
        y = torch.tanh(uy @ self.E_y) * self.scale
        scores = self.torus_f(x, y) + self.bias
        topk_s, topk_i = torch.topk(scores, self.top_k, dim=-1)
        probs = F.softmax(scores, dim=-1)
        aux = (probs.mean(0)**2).sum() * self.num_experts
        return topk_i, topk_s, scores, aux

class Model(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.router = TorusRouter(d_model, num_experts, top_k)
        self.head   = nn.Linear(num_experts, 4)

    def forward(self, u):
        idx, s, all_s, aux = self.router(u)
        return self.head(all_s), aux, idx

def train():
    print("="*60)
    print("  4CM-MoE TorusRouter — TF-IDF Language Training")
    print("="*60)

    sents, labs = [], []
    for name, ss in TOPICS.items():
        sents.extend(ss)
        labs.extend([TOPIC_LABELS[name]]*len(ss))

    vec  = TfidfVectorizer(max_features=256)
    arr  = vec.fit_transform(sents).toarray()
    X    = torch.tensor(arr, dtype=torch.float32)
    Y    = torch.tensor(labs)
    d    = X.shape[1]
    print(f"\nData: {X.shape} | Labels: {Y.shape}")

    num_experts = 8
    model = Model(d, num_experts=num_experts, top_k=2)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Step':>6} {'CE':>10} {'Aux':>10} {'Acc':>8} {'c':>8} {'d':>8}")
    print("-"*60)

    for step in range(300):
        logits, aux, idx = model(X)
        ce   = F.cross_entropy(logits, Y)
        loss = ce + 0.01 * aux
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 30 == 0:
            acc = (logits.argmax(-1)==Y).float().mean().item()
            print(f"{step:>6} {ce.item():>10.4f} {aux.item():>10.4f} "
                  f"{acc:>8.2%} {model.router.c.item():>8.4f} {model.router.d.item():>8.4f}")

    print("\n"+"="*60)
    print("  Expert Selection Pattern by Topic")
    print("="*60)
    model.eval()
    with torch.no_grad():
        logits, _, idx = model(X)
        acc = (logits.argmax(-1)==Y).float().mean().item()
    print(f"\nFinal Accuracy: {acc:.2%}\n")

    for name in TOPIC_NAMES:
        lbl   = TOPIC_LABELS[name]
        mask  = (Y == lbl)
        t_idx = idx[mask]
        cnt   = torch.zeros(num_experts)
        for e in t_idx.flatten(): cnt[e] += 1
        top3  = cnt.topk(3)
        s     = " | ".join([f"E{i.item()}({int(c.item())} count)"
                            for i,c in zip(top3.indices, top3.values)])
        print(f"  {name:<12} {s}")

    print("\n"+"="*60)
    print("  Final Torus Params (learned):")
    print("="*60)
    r = model.router
    print(f"  c  = {r.c.item():.4f}  (init 2.0)")
    print(f"  d  = {r.d.item():.4f}  (init 2.0)")
    print(f"  a1 = {r.a1.item():.4f} (init 4.0)")
    print(f"  b1 = {r.b1.item():.4f} (init 4.0)")

if __name__ == "__main__":
    train()