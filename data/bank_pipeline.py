import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE      = r"C:\Users\prane\Desktop\VIT\Projects\Prob\data3"
RAW_CSV   = os.path.join(BASE, "bank-additional-full.csv")
OUT_PRE   = os.path.join(BASE, "bank_preprocessed.csv")
OUT_SMO   = os.path.join(BASE, "bank_smote.csv")
PLOTS_RAW = os.path.join(BASE, "plots_raw")
PLOTS_PRE = os.path.join(BASE, "plots_preprocessed")
PLOTS_SMO = os.path.join(BASE, "plots_smote")
for d in [PLOTS_RAW, PLOTS_PRE, PLOTS_SMO]:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
})
PAL = {
    "blue":"#2563eb","green":"#10b981","red":"#ef4444","amber":"#f59e0b",
    "purple":"#8b5cf6","pink":"#ec4899","teal":"#0d9488","orange":"#f97316",
    "slate":"#64748b","indigo":"#6366f1",
}

def save_fig(fig, path, name):
    fig.savefig(os.path.join(path, name), bbox_inches="tight", dpi=200)
    plt.close(fig)

# ============================================================
# STEP 1 — Load Raw Data
# ============================================================
print("="*70)
print("STEP 1: Loading raw data ...")
df_raw = pd.read_csv(RAW_CSV, sep=";")
N = len(df_raw)
print(f"  Raw shape: {df_raw.shape}  |  Target: {dict(df_raw['y'].value_counts())}")

# ============================================================
# STEP 2 — RAW EDA PLOTS  (15 plots -> plots_raw/)
# ============================================================
print("\nSTEP 2: Generating raw EDA plots ...")
df_r = df_raw.copy()

# 01 Age (5-yr bins)
fig, ax = plt.subplots(figsize=(12,5))
bins_a = np.arange(15,105,5)
labels_a = [f"{b}-{b+4}" for b in bins_a[:-1]]
cnt_a, _ = np.histogram(df_r["age"].dropna(), bins=bins_a)
bars = ax.bar(range(len(cnt_a)), cnt_a, color=PAL["blue"], edgecolor="#1e3a8a", width=0.85)
ax.set_xticks(range(len(labels_a))); ax.set_xticklabels(labels_a, rotation=45, ha="right")
ax.set_title("Age Distribution in 5-Year Classes (Raw Data)", fontweight="bold")
ax.set_xlabel("Age Group"); ax.set_ylabel("Count"); ax.set_ylim(0, max(cnt_a)*1.18)
for bar, v in zip(bars, cnt_a):
    if v > 0:
        ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold")
save_fig(fig, PLOTS_RAW, "01_age_distribution.png")

# 02 Job
fig, ax = plt.subplots(figsize=(12,5))
jc = df_r["job"].value_counts()
cm = plt.cm.get_cmap("tab20", len(jc))
bars = ax.bar(jc.index, jc.values, color=[cm(i) for i in range(len(jc))], edgecolor="black", width=0.7)
ax.set_title("Job Type Distribution (Raw Data)", fontweight="bold")
ax.set_ylabel("Count"); ax.set_ylim(0, jc.max()*1.18); plt.xticks(rotation=35, ha="right")
for bar, v in zip(bars, jc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
save_fig(fig, PLOTS_RAW, "02_job_distribution.png")

# 03 Marital
fig, ax = plt.subplots(figsize=(7,5))
mc = df_r["marital"].value_counts()
bars = ax.bar(mc.index, mc.values, color=[PAL["blue"],PAL["purple"],PAL["green"],PAL["slate"]][:len(mc)], edgecolor="black", width=0.5)
ax.set_title("Marital Status Distribution (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, mc.max()*1.15)
for bar, v in zip(bars, mc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "03_marital_distribution.png")

# 04 Education
fig, ax = plt.subplots(figsize=(10,5))
ec = df_r["education"].value_counts()
clrs4 = [PAL["blue"],PAL["green"],PAL["amber"],PAL["red"],PAL["purple"],PAL["teal"],PAL["pink"],PAL["slate"]]
bars = ax.bar(ec.index, ec.values, color=clrs4[:len(ec)], edgecolor="black", width=0.6)
ax.set_title("Education Level Distribution (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, ec.max()*1.15)
plt.xticks(rotation=35, ha="right")
for bar, v in zip(bars, ec.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
save_fig(fig, PLOTS_RAW, "04_education_distribution.png")

# 05 Default/Housing/Loan panel
fig, axes = plt.subplots(1,3, figsize=(14,5))
for ax, col, title in zip(axes, ["default","housing","loan"],
                           ["Credit Default","Housing Loan","Personal Loan"]):
    c = df_r[col].value_counts()
    clrs = [PAL["green"] if x=="no" else PAL["red"] if x=="yes" else PAL["slate"] for x in c.index]
    bars = ax.bar(c.index, c.values, color=clrs, edgecolor="black", width=0.5)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, c.max()*1.18)
    for bar, v in zip(bars, c.values):
        ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)
plt.suptitle("Financial Flags Distribution (Raw Data)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, PLOTS_RAW, "05_financial_flags_distribution.png")

# 06 Contact type
fig, ax = plt.subplots(figsize=(7,5))
ctc = df_r["contact"].value_counts()
bars = ax.bar(ctc.index, ctc.values, color=[PAL["teal"],PAL["orange"]][:len(ctc)], edgecolor="black", width=0.45)
ax.set_title("Contact Type (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, ctc.max()*1.15)
for bar, v in zip(bars, ctc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "06_contact_type_distribution.png")

# 07 Month
fig, ax = plt.subplots(figsize=(11,5))
mo_ord = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
moc = df_r["month"].value_counts().reindex([m for m in mo_ord if m in df_r["month"].unique()])
cm7 = plt.cm.get_cmap("coolwarm", len(moc))
bars = ax.bar(moc.index, moc.values, color=[cm7(i/len(moc)) for i in range(len(moc))], edgecolor="black", width=0.7)
ax.set_title("Campaign Month Distribution (Raw Data)", fontweight="bold"); ax.set_xlabel("Month"); ax.set_ylabel("Count"); ax.set_ylim(0, moc.max()*1.15)
for bar, v in zip(bars, moc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
save_fig(fig, PLOTS_RAW, "07_month_distribution.png")

# 08 Day of week
fig, ax = plt.subplots(figsize=(8,5))
dowc = df_r["day_of_week"].value_counts().reindex(["mon","tue","wed","thu","fri"])
bars = ax.bar(dowc.index, dowc.values, color=[PAL["blue"],PAL["green"],PAL["amber"],PAL["purple"],PAL["teal"]], edgecolor="black", width=0.55)
ax.set_title("Day of Week Distribution (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, dowc.max()*1.15)
for bar, v in zip(bars, dowc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "08_day_of_week_distribution.png")

# 09 Poutcome
fig, ax = plt.subplots(figsize=(8,5))
poc = df_r["poutcome"].value_counts()
poclr = {"nonexistent":PAL["slate"],"failure":PAL["red"],"success":PAL["green"]}
bars = ax.bar(poc.index, poc.values, color=[poclr.get(x,PAL["blue"]) for x in poc.index], edgecolor="black", width=0.5)
ax.set_title("Previous Campaign Outcome (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, poc.max()*1.15)
for bar, v in zip(bars, poc.values):
    ax.annotate(f"{v:,}\n({v/N*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "09_poutcome_distribution.png")

# 10 Target y
fig, ax = plt.subplots(figsize=(7,5))
yc = df_r["y"].value_counts()
bars = ax.bar(["No (Did Not Subscribe)","Yes (Subscribed)"], [yc.get("no",0), yc.get("yes",0)],
               color=[PAL["blue"],PAL["red"]], edgecolor="black", width=0.45)
ax.set_title("Target: Term Deposit Subscription (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, max(yc.values)*1.15)
for bar, v in zip(bars, [yc.get("no",0), yc.get("yes",0)]):
    ax.annotate(f"{v:,}\n({v/N*100:.2f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "10_target_distribution.png")

# 11 Duration
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,7), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
dur = df_r["duration"]
sns.histplot(dur, kde=True, ax=ax1, color=PAL["blue"], bins=50, edgecolor="black")
ax1.axvline(dur.mean(), color=PAL["green"], linestyle="--", lw=2, label=f"Mean: {dur.mean():.0f}s")
ax1.axvline(dur.median(), color=PAL["amber"], linestyle="-", lw=2, label=f"Median: {dur.median():.0f}s")
ax1.text(0.68,0.55, f"Min: {dur.min()}s | Max: {dur.max()}s\nMean: {dur.mean():.0f}s | Median: {dur.median():.0f}s\nStd: {dur.std():.0f}s | Skew: {dur.skew():.3f}",
         transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95), fontsize=9, family="monospace")
ax1.set_title("Call Duration Distribution (Raw Data)", fontweight="bold"); ax1.legend()
sns.boxplot(x=dur, ax=ax2, color="#60a5fa"); ax2.set_xlabel("Call Duration (seconds)")
save_fig(fig, PLOTS_RAW, "11_duration_distribution.png")

# 12 Campaign contacts
fig, ax = plt.subplots(figsize=(10,5))
camp = df_r["campaign"].clip(upper=15)
campc = camp.value_counts().sort_index()
bars = ax.bar(campc.index.astype(str), campc.values, color=PAL["purple"], edgecolor="black", width=0.75)
ax.set_title("Campaign Contacts (Raw Data, display-capped at 15)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, campc.max()*1.15)
for bar, v in zip(bars, campc.values):
    if v > 200:
        ax.annotate(f"{v:,}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
save_fig(fig, PLOTS_RAW, "12_campaign_contacts_distribution.png")

# 13 Macro panel
macro5 = ["emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
titles5 = ["Emp. Variation Rate","Consumer Price Idx","Consumer Confidence Idx","Euribor 3M","Nr. Employees"]
clrs5 = [PAL["blue"],PAL["green"],PAL["red"],PAL["amber"],PAL["purple"]]
fig, axes = plt.subplots(1,5, figsize=(20,5))
for ax, col, t, c in zip(axes, macro5, titles5, clrs5):
    sns.histplot(df_r[col], kde=True, ax=ax, color=c, bins=30, edgecolor="black")
    ax.set_title(t, fontweight="bold", fontsize=10)
    ax.axvline(df_r[col].mean(), color="black", linestyle="--", lw=1.5, label=f"mu={df_r[col].mean():.2f}")
    ax.legend(fontsize=8)
plt.suptitle("Macroeconomic Indicator Distributions (Raw Data)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, PLOTS_RAW, "13_macroeconomic_distributions.png")

# 14 Correlation heatmap raw
num14 = df_r[["age","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]].copy()
num14["y_enc"] = (df_r["y"]=="yes").astype(int)
fig, ax = plt.subplots(figsize=(11,9))
sns.heatmap(num14.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size":9}, cbar_kws={"shrink":.8}, ax=ax)
ax.set_title("Correlation Heatmap — Numeric Features (Raw Data)", fontsize=13, fontweight="bold", pad=15)
save_fig(fig, PLOTS_RAW, "14_correlation_heatmap_raw.png")

# 15 Unknown summary
unk_cl = ["job","marital","education","default","housing","loan"]
unk_ct = {c: (df_r[c].astype(str)=="unknown").sum() for c in unk_cl}
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(unk_ct.keys(), unk_ct.values(),
              color=[PAL["red"],PAL["amber"],PAL["orange"],PAL["purple"],PAL["green"],PAL["teal"]],
              edgecolor="black", width=0.55)
ax.set_title("'Unknown' Value Counts per Column (Raw Data)", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, max(unk_ct.values())*1.22)
for bar, c in zip(bars, unk_cl):
    v = unk_ct[c]
    ax.annotate(f"{v:,}\n({v/N*100:.2f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
save_fig(fig, PLOTS_RAW, "15_unknown_values_summary.png")
print(f"  Raw EDA: 15 plots -> {PLOTS_RAW}")

# ============================================================
# STEP 3 — PREPROCESSING
# ============================================================
print("\nSTEP 3: Preprocessing ...")
df = df_raw.copy()

# 3a target
df["y"] = (df["y"]=="yes").astype(int)
print(f"  3a target encoded: {dict(df['y'].value_counts())}")

# 3b unknowns -> NaN
for c in ["job","marital","education","default","housing","loan"]:
    n = (df[c]=="unknown").sum()
    df[c] = df[c].replace("unknown", np.nan)
    print(f"  3b {c}: {n} unknowns -> NaN")

# 3c pdays 999 -> NaN
df["pdays_contacted"] = (df["pdays"] != 999).astype(int)
df["pdays"] = df["pdays"].replace(999, np.nan)
print("  3c pdays 999 -> NaN | pdays_contacted flag created")

# 3d binary encode
df["contact"] = (df["contact"]=="cellular").astype(int)
df["housing"] = df["housing"].map({"yes":1,"no":0})
df["loan"]    = df["loan"].map({"yes":1,"no":0})
df["default"] = df["default"].map({"yes":1,"no":0})
print("  3d binary encoding: contact, housing, loan, default")

# 3e education ordinal
edu_ord = {"illiterate":0,"basic.4y":1,"basic.6y":2,"basic.9y":3,"high.school":4,"professional.course":5,"university.degree":6}
df["education_ordinal"] = df["education"].map(edu_ord)
df.drop(columns=["education"], inplace=True)
print("  3e education ordinal encoded 0-6")

# 3f feature engineering
df["campaign_capped"] = df["campaign"].clip(upper=10)
df["was_previously_contacted"] = df["pdays_contacted"]
df["contact_frequency_ratio"] = df["previous"] / (df["campaign"] + 1)
_sc = StandardScaler()
_macro = df[["emp.var.rate","euribor3m","nr.employed"]].values
df["economic_pressure_index"] = _sc.fit_transform(_macro).mean(axis=1)
df["age_bucket"] = pd.cut(df["age"], bins=[16,25,35,50,65,120], labels=[0,1,2,3,4]).astype(float)
df["duration_bucket"] = pd.cut(df["duration"], bins=[-1,60,300,5000], labels=[0,1,2]).astype(int)
mq_map = {"jan":1,"feb":1,"mar":1,"apr":2,"may":2,"jun":2,"jul":3,"aug":3,"sep":3,"oct":4,"nov":4,"dec":4}
df["month_quarter"] = df["month"].map(mq_map)
print("  3f feature engineering done (7 new features)")

# 3g KNN imputation on numeric NaNs
knn_cols = ["housing","loan","default","education_ordinal","pdays","age_bucket"]
knn_imp = KNNImputer(n_neighbors=7, weights="distance")
df[knn_cols] = knn_imp.fit_transform(df[knn_cols])
for c in ["housing","loan","default","education_ordinal","age_bucket"]:
    df[c] = df[c].round().astype(int)
print("  3g KNN imputation done")

# 3h OHE
df["job"]     = df["job"].fillna(df["job"].mode()[0])
df["marital"] = df["marital"].fillna(df["marital"].mode()[0])
df = pd.get_dummies(df, columns=["job","marital","month","day_of_week","poutcome"],
                    prefix=["job","marital","month","dow","poutcome"], dtype=int)
df.drop(columns=["campaign"], inplace=True)
print(f"  3h OHE done | shape: {df.shape}")

# 3i Yeo-Johnson
cont_cols = ["age","duration","pdays","previous","emp.var.rate","cons.price.idx",
             "cons.conf.idx","euribor3m","nr.employed","campaign_capped",
             "contact_frequency_ratio","economic_pressure_index"]
pt = PowerTransformer(method="yeo-johnson", standardize=True)
norm_names = [f"{c}_normalized" for c in cont_cols]
df[norm_names] = pt.fit_transform(df[cont_cols])
print(f"  3i Yeo-Johnson on {len(cont_cols)} features")

all_feat = [c for c in df.columns if c != "y"]
df = df[all_feat + ["y"]].copy()
df.to_csv(OUT_PRE, index=False)
print(f"\n  Saved bank_preprocessed.csv | {df.shape[0]} rows x {df.shape[1]} cols")
print(f"  Target: {dict(df['y'].value_counts())}")

# ============================================================
# STEP 4 — SMOTE
# ============================================================
print("\nSTEP 4: SMOTE ...")
X = df.drop(columns=["y"]); y_col = df["y"]
print(f"  Before SMOTE: {dict(y_col.value_counts())}")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y_col)
df_smote = pd.concat([pd.DataFrame(X_sm, columns=X.columns), pd.Series(y_sm, name="y")], axis=1)
for c in df_smote.columns:
    try:
        if df_smote[c].nunique() < 60 and (df_smote[c] % 1 == 0).all():
            df_smote[c] = df_smote[c].astype(int)
    except Exception:
        pass
df_smote.to_csv(OUT_SMO, index=False)
print(f"  After SMOTE: {dict(df_smote['y'].value_counts())}")
print(f"  Saved bank_smote.csv | {df_smote.shape[0]} rows x {df_smote.shape[1]} cols")

# ============================================================
# STEP 5 — 20 FINAL PLOTS PER DATASET
# ============================================================
print("\nSTEP 5: Generating 20 plots per final dataset ...")

def gen_plots(dff, out_dir, dname):
    os.makedirs(out_dir, exist_ok=True)
    tn = len(dff)
    print(f"  {dname} ({tn} rows) -> {out_dir}")
    def sp(fig, fn): save_fig(fig, out_dir, fn)

    # P01 Age
    fig, ax = plt.subplots(figsize=(12,5))
    bins_a = np.arange(15,105,5); lbs_a = [f"{b}-{b+4}" for b in bins_a[:-1]]
    cnt_a, _ = np.histogram(dff["age"].dropna(), bins=bins_a)
    bars = ax.bar(range(len(cnt_a)), cnt_a, color=PAL["blue"], edgecolor="#1e3a8a", width=0.85)
    ax.set_xticks(range(len(lbs_a))); ax.set_xticklabels(lbs_a, rotation=45, ha="right")
    ax.set_title(f"Age Distribution 5-Year Classes ({dname})", fontweight="bold")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Count"); ax.set_ylim(0, max(cnt_a)*1.18)
    for bar, v in zip(bars, cnt_a):
        if v>0:
            ax.annotate(f"{v:,}\n({v/tn*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold")
    sp(fig, "01_age_distribution.png")

    # P02 Target
    fig, ax = plt.subplots(figsize=(7,5))
    yc = dff["y"].value_counts().sort_index()
    yp = yc/tn*100
    bars = ax.bar(["0 (No Subscription)","1 (Subscribed)"], yc.values, color=[PAL["blue"],PAL["red"]], edgecolor="black", width=0.45)
    ax.set_title(f"Target: Term Deposit Subscription ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, yc.max()*1.18)
    for bar, v, p in zip(bars, yc.values, yp.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "02_target_distribution.png")

    # P03 Contact
    fig, ax = plt.subplots(figsize=(7,5))
    ctc = dff["contact"].value_counts().sort_index()
    bars = ax.bar(["0 (Telephone)","1 (Cellular)"], ctc.values, color=[PAL["orange"],PAL["teal"]], edgecolor="black", width=0.45)
    ax.set_title(f"Contact Type ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, ctc.max()*1.15)
    for bar, v in zip(bars, ctc.values):
        ax.annotate(f"{v:,}\n({v/tn*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "03_contact_type_distribution.png")

    # P04 Financial flags
    fig, axes = plt.subplots(1,3, figsize=(14,5))
    for ax, col, lbl in zip(axes, ["housing","loan","default"], ["Housing Loan","Personal Loan","Credit Default"]):
        c = dff[col].value_counts().sort_index(); p = c/tn*100
        bars = ax.bar(["0 (No)","1 (Yes)"][:len(c)], c.values, color=[PAL["green"],PAL["red"]][:len(c)], edgecolor="black", width=0.45)
        ax.set_title(f"{lbl} ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, c.max()*1.18)
        for bar, v, pct in zip(bars, c.values, p.values):
            ax.annotate(f"{v:,}\n({pct:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)
    plt.suptitle(f"Financial Flags ({dname})", fontsize=13, fontweight="bold", y=1.02); plt.tight_layout()
    sp(fig, "04_financial_flags_distribution.png")

    # P05 Age Bucket
    fig, ax = plt.subplots(figsize=(9,5))
    abc = dff["age_bucket"].value_counts().sort_index()
    abp = abc/tn*100
    ab_lbs = ["17-25 (Youth)","26-35 (Young Adult)","36-50 (Middle Age)","51-65 (Senior)","65+ (Elder)"]
    ab_cl = [PAL["green"],PAL["blue"],PAL["purple"],PAL["amber"],PAL["red"]]
    bars = ax.bar([ab_lbs[int(i)] for i in abc.index], abc.values, color=[ab_cl[int(i)] for i in abc.index], edgecolor="black", width=0.55)
    ax.set_title(f"Age Bucket Distribution ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, abc.max()*1.15)
    plt.xticks(rotation=25, ha="right")
    for bar, v, p in zip(bars, abc.values, abp.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "05_age_bucket_distribution.png")

    # P06 Duration Bucket
    fig, ax = plt.subplots(figsize=(8,5))
    dbc = dff["duration_bucket"].value_counts().sort_index(); dbp = dbc/tn*100
    db_lbs = ["Short (<60s)","Medium (60-300s)","Long (>300s)"]
    bars = ax.bar([db_lbs[i] for i in dbc.index], dbc.values,
                  color=[PAL["green"],PAL["blue"],PAL["purple"]][:len(dbc)], edgecolor="black", width=0.45)
    ax.set_title(f"Call Duration Bucket ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, dbc.max()*1.15)
    for bar, v, p in zip(bars, dbc.values, dbp.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "06_duration_bucket_distribution.png")

    # P07 Previously contacted
    fig, ax = plt.subplots(figsize=(7,5))
    pcc = dff["was_previously_contacted"].value_counts().sort_index(); pcp = pcc/tn*100
    bars = ax.bar(["0 (First Time)","1 (Repeat Contact)"], pcc.values, color=[PAL["slate"],PAL["teal"]], edgecolor="black", width=0.45)
    ax.set_title(f"Previously Contacted Flag ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, pcc.max()*1.15)
    for bar, v, p in zip(bars, pcc.values, pcp.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "07_previously_contacted_distribution.png")

    # P08 Education ordinal
    fig, ax = plt.subplots(figsize=(10,5))
    educ = dff["education_ordinal"].value_counts().sort_index(); edup = educ/tn*100
    edu_lbs = ["Illiterate (0)","Basic 4yr (1)","Basic 6yr (2)","Basic 9yr (3)","High School (4)","Professional (5)","University (6)"]
    edu_cl = [PAL["red"],PAL["orange"],PAL["amber"],PAL["green"],PAL["teal"],PAL["blue"],PAL["purple"]]
    bars = ax.bar([edu_lbs[i] for i in educ.index], educ.values, color=[edu_cl[i] for i in educ.index], edgecolor="black", width=0.65)
    ax.set_title(f"Education Level Ordinal ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, educ.max()*1.15)
    plt.xticks(rotation=30, ha="right")
    for bar, v, p in zip(bars, educ.values, edup.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    sp(fig, "08_education_ordinal_distribution.png")

    # P09 Month Quarter
    fig, ax = plt.subplots(figsize=(8,5))
    mqc = dff["month_quarter"].value_counts().sort_index(); mqp = mqc/tn*100
    bars = ax.bar([f"Q{int(i)}" for i in mqc.index], mqc.values,
                  color=[PAL["green"],PAL["blue"],PAL["amber"],PAL["red"]][:len(mqc)], edgecolor="black", width=0.5)
    ax.set_title(f"Campaign Quarter ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, mqc.max()*1.15)
    for bar, v, p in zip(bars, mqc.values, mqp.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "09_month_quarter_distribution.png")

    # P10 Campaign capped
    fig, ax = plt.subplots(figsize=(10,5))
    ccc = dff["campaign_capped"].value_counts().sort_index(); ccp = ccc/tn*100
    bars = ax.bar(ccc.index.astype(str), ccc.values, color=PAL["purple"], edgecolor="black", width=0.75)
    ax.set_title(f"Campaign Contacts Capped at 10 ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, ccc.max()*1.15)
    for bar, v, p in zip(bars, ccc.values, ccp.values):
        if v > 150:
            ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    sp(fig, "10_campaign_capped_distribution.png")

    # P11 Duration continuous
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,7), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    dur2 = dff["duration"]
    sns.histplot(dur2, kde=True, ax=ax1, color=PAL["blue"], bins=40, edgecolor="black")
    ax1.axvline(dur2.mean(), color=PAL["green"], linestyle="--", lw=2, label=f"Mean: {dur2.mean():.0f}s")
    ax1.axvline(dur2.median(), color=PAL["amber"], linestyle="-", lw=2, label=f"Median: {dur2.median():.0f}s")
    ax1.text(0.68,0.55, f"Min: {dur2.min():.0f}s|Max: {dur2.max():.0f}s\nMean: {dur2.mean():.0f}s|Median: {dur2.median():.0f}s\nStd: {dur2.std():.0f}s|Skew: {dur2.skew():.3f}",
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95), fontsize=9, family="monospace")
    ax1.set_title(f"Call Duration ({dname})", fontweight="bold"); ax1.legend()
    sns.boxplot(x=dur2, ax=ax2, color="#60a5fa"); ax2.set_xlabel("Duration (s)")
    sp(fig, "11_duration_distribution.png")

    # P12 Economic Pressure Index
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    epi = dff["economic_pressure_index"]
    sns.histplot(epi, kde=True, ax=ax1, color=PAL["red"], bins=35, edgecolor="black")
    ax1.axvline(epi.mean(), color=PAL["green"], linestyle="--", lw=2, label=f"Mean: {epi.mean():.3f}")
    ax1.set_title(f"Economic Pressure Composite Index ({dname})", fontweight="bold"); ax1.legend()
    sns.boxplot(x=epi, ax=ax2, color="#fca5a5"); ax2.set_xlabel("Economic Pressure Index")
    sp(fig, "12_economic_pressure_index_distribution.png")

    # P13 Contact Frequency Ratio
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    cfr = dff["contact_frequency_ratio"]
    sns.histplot(cfr, kde=True, ax=ax1, color=PAL["teal"], bins=35, edgecolor="black")
    ax1.axvline(cfr.mean(), color=PAL["red"], linestyle="--", lw=2, label=f"Mean: {cfr.mean():.3f}")
    ax1.set_title(f"Contact Frequency Ratio ({dname})", fontweight="bold"); ax1.legend()
    sns.boxplot(x=cfr, ax=ax2, color="#5eead4"); ax2.set_xlabel("previous / campaign")
    sp(fig, "13_contact_frequency_ratio_distribution.png")

    # P14 Pdays
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,6), gridspec_kw={"height_ratios":[3,1]}, sharex=True)
    pdd = dff["pdays"].dropna()
    if len(pdd) > 10:
        sns.histplot(pdd, kde=True, ax=ax1, color=PAL["purple"], bins=30, edgecolor="black")
        ax1.axvline(pdd.mean(), color=PAL["green"], linestyle="--", lw=2, label=f"Mean: {pdd.mean():.1f}d")
        ax1.set_title(f"Days Since Last Contact — pdays ({dname})", fontweight="bold"); ax1.legend()
        sns.boxplot(x=pdd, ax=ax2, color="#c4b5fd"); ax2.set_xlabel("Days Since Last Contact")
    sp(fig, "14_pdays_distribution.png")

    # P15 Yeo-Johnson normalized
    norm_here = [c for c in norm_names if c in dff.columns][:4]
    if norm_here:
        fig, axes = plt.subplots(1, len(norm_here), figsize=(4*len(norm_here),5))
        if not hasattr(axes, "__iter__"): axes = [axes]
        nc_cl = [PAL["blue"],PAL["green"],PAL["red"],PAL["amber"]]
        for ax, col, c in zip(axes, norm_here, nc_cl):
            sns.histplot(dff[col], kde=True, ax=ax, color=c, bins=30, edgecolor="black")
            ax.set_title(col.replace("_normalized","").replace("_"," ").title()+f"\n(Skew: {dff[col].skew():.2f})", fontsize=9.5, fontweight="bold")
        plt.suptitle(f"Yeo-Johnson Normalized Features ({dname})", fontsize=12, fontweight="bold", y=1.03); plt.tight_layout()
        sp(fig, "15_normalized_features_distribution.png")

    # P16 Job OHE
    jcols = [c for c in dff.columns if c.startswith("job_")]
    if jcols:
        js = {c.replace("job_",""):int(dff[c].sum()) for c in jcols}
        fig, ax = plt.subplots(figsize=(12,5))
        cm16 = plt.cm.get_cmap("tab20", len(js))
        bars = ax.bar(js.keys(), js.values(), color=[cm16(i) for i in range(len(js))], edgecolor="black", width=0.7)
        ax.set_title(f"Job Category ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, max(js.values())*1.18)
        plt.xticks(rotation=35, ha="right")
        for bar, v in zip(bars, js.values()):
            ax.annotate(f"{v:,}\n({v/tn*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        sp(fig, "16_job_distribution.png")

    # P17 Marital OHE
    mrcols = [c for c in dff.columns if c.startswith("marital_")]
    if mrcols:
        mrs = {c.replace("marital_",""):int(dff[c].sum()) for c in mrcols}
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(mrs.keys(), mrs.values(), color=[PAL["blue"],PAL["purple"],PAL["green"],PAL["amber"]][:len(mrs)], edgecolor="black", width=0.5)
        ax.set_title(f"Marital Status ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, max(mrs.values())*1.15)
        for bar, v in zip(bars, mrs.values()):
            ax.annotate(f"{v:,}\n({v/tn*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        sp(fig, "17_marital_distribution.png")

    # P18 Poutcome OHE
    pocols = [c for c in dff.columns if c.startswith("poutcome_")]
    if pocols:
        pos = {c.replace("poutcome_",""):int(dff[c].sum()) for c in pocols}
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(pos.keys(), pos.values(), color=[PAL["slate"],PAL["red"],PAL["green"]][:len(pos)], edgecolor="black", width=0.5)
        ax.set_title(f"Previous Campaign Outcome ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, max(pos.values())*1.15)
        for bar, v in zip(bars, pos.values()):
            ax.annotate(f"{v:,}\n({v/tn*100:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        sp(fig, "18_poutcome_distribution.png")

    # P19 Pdays contacted flag
    fig, ax = plt.subplots(figsize=(7,5))
    p19c = dff["pdays_contacted"].value_counts().sort_index(); p19p = p19c/tn*100
    bars = ax.bar(["0 (Never Contacted Before)","1 (Was Contacted Before)"], p19c.values, color=[PAL["slate"],PAL["teal"]], edgecolor="black", width=0.45)
    ax.set_title(f"Prior Campaign Contact Flag ({dname})", fontweight="bold"); ax.set_ylabel("Count"); ax.set_ylim(0, p19c.max()*1.15)
    for bar, v, p in zip(bars, p19c.values, p19p.values):
        ax.annotate(f"{v:,}\n({p:.1f}%)", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    sp(fig, "19_pdays_contacted_flag_distribution.png")

    # P20 Full correlation heatmap
    num_c20 = dff.select_dtypes(include=[np.number]).columns.tolist()
    corr20 = dff[num_c20].corr()
    nf = len(num_c20)
    fig, ax = plt.subplots(figsize=(max(14,nf*0.55), max(11,nf*0.45)))
    sns.heatmap(corr20, annot=(nf<=35), fmt=".2f" if nf<=35 else "",
                cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.3 if nf>30 else 0.5,
                annot_kws={"size":6} if nf>30 else {"size":8},
                cbar_kws={"shrink":.7}, ax=ax)
    ax.set_title(f"Correlation Heatmap — All Features ({dname})", fontsize=13, fontweight="bold", pad=15)
    sp(fig, "20_correlation_heatmap.png")
    print(f"    20 plots saved for {dname}")

gen_plots(df, PLOTS_PRE, "bank_preprocessed")
gen_plots(df_smote, PLOTS_SMO, "bank_smote")

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"  Raw          : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
print(f"  Preprocessed : {df.shape[0]} rows x {df.shape[1]} cols -> {OUT_PRE}")
print(f"  SMOTE        : {df_smote.shape[0]} rows x {df_smote.shape[1]} cols -> {OUT_SMO}")
print(f"  plots_raw/   : 15 plots")
print(f"  plots_preprocessed/ : 20 plots")
print(f"  plots_smote/ : 20 plots")
print("="*70)
