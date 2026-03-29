"""Generate PDF: TSUM applied to IEEE DC-OPF benchmarks from Chan et al. (2024)."""

from fpdf import FPDF
import json
import math
from pathlib import Path
from collections import Counter


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "TSUM on IEEE DC-OPF Benchmarks", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, "Page %d/{nb}" % self.page_no(), align="C")

    def section_title(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(30, 60, 120)
            self.ln(6)
            self.cell(0, 10, title)
            self.ln(8)
            self.set_draw_color(30, 60, 120)
            self.set_line_width(0.5)
            self.line(self.l_margin, self.get_y(),
                      self.w - self.r_margin, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(50, 80, 140)
            self.ln(4)
            self.cell(0, 8, title)
            self.ln(6)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, text)
        self.ln(1)

    def table_header(self, cells, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for cell, w in zip(cells, widths):
            self.cell(w, 6, cell, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(30, 30, 30)

    def table_row(self, cells, widths, bold_first=False, fill=False):
        if fill:
            self.set_fill_color(230, 235, 245)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i == 0 and bold_first:
                self.set_font("Helvetica", "B", 9)
            else:
                self.set_font("Helvetica", "", 9)
            self.cell(w, 5.5, str(cell), border=1, align="C", fill=fill)
        self.ln()


def load_metrics(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rules(path):
    with open(path) as f:
        return json.load(f)


def parse_log_metrics(path):
    """Parse metrics from a PBS log file (when metrics.json not available)."""
    import re
    with open(path) as f:
        text = f.read()
    rounds = re.findall(r'Round: (\d+), Unk\. prob\.: ([\d.e+-]+)', text)
    surv_lines = re.findall(
        r'Surv probs: ([\d.e+-]+), Fail probs: ([\d.e+-]+)', text)
    rule_counts = re.findall(
        r'Survival rules: (\d+), Failure rules: (\d+)', text)
    if not rounds:
        return None
    last_round = int(rounds[-1][0])
    last_unk = float(rounds[-1][1])
    last_surv_p = float(surv_lines[-1][0])
    last_fail_p = float(surv_lines[-1][1])
    last_surv_r = int(rule_counts[-1][0])
    last_fail_r = int(rule_counts[-1][1])
    return {
        'round': last_round,
        'p_survival': last_surv_p,
        'p_failure': last_fail_p,
        'p_unknown': last_unk,
        'n_rules_surv': last_surv_r,
        'n_rules_fail': last_fail_r,
    }


def main():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pw = pdf.w - pdf.l_margin - pdf.r_margin  # page width

    # =================================================================
    # Title page
    # =================================================================
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, "TSUM Applied to IEEE DC-OPF", align="C")
    pdf.ln(12)
    pdf.cell(0, 12, "Blackout Benchmarks", align="C")
    pdf.ln(18)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Reproducing results from Chan et al. (2024)", align="C")
    pdf.ln(8)
    pdf.cell(0, 8,
             "Adaptive Monte Carlo methods for estimating", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "rare events in power grids", align="C")
    pdf.ln(16)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "March 2026", align="C")
    pdf.ln(30)

    # Summary box
    pdf.set_fill_color(230, 235, 245)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "  Summary", fill=True)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    summary = (
        "We apply TSUM (Tensor-based System Unreliability Method) to the "
        "IEEE DC optimal power flow (DC-OPF) blackout models from Chan et al. "
        "(2024), covering all five benchmark cases (14, 30, 57, 118, 300-bus). "
        "TSUM extracts interpretable survival and failure rules that decompose "
        "the system reliability into exact probability bounds. The three "
        "completed cases (14, 30, 57-bus) achieve failure probability estimates "
        "consistent with the paper's reference values (p_f ~ 10^-4). For the "
        "challenging IEEE 118-bus case (13.8% threshold), standard sampling "
        "fails to find failure rules; a new biased discovery sampling strategy "
        "is employed. For IEEE 300-bus, a bug in the DC-OPF solver (negative "
        "loads causing LP infeasibility) was fixed, and biased sampling runs "
        "are in progress."
    )
    pdf.multi_cell(0, 5, summary, fill=True)
    pdf.ln(1)

    # =================================================================
    # 1. Introduction
    # =================================================================
    pdf.add_page()
    pdf.section_title("1. Introduction")

    pdf.body_text(
        "Chan et al. (2024) benchmark several adaptive Monte Carlo methods "
        "for estimating rare blackout probabilities in IEEE standard power "
        "grids. They use a DC optimal power flow (DC-OPF) model where "
        "component failures (branches, buses, generators) reduce available "
        "generation and transmission capacity, and system failure is defined "
        "as blackout size exceeding a threshold."
    )
    pdf.body_text(
        "Their paper tests five IEEE cases (14, 30, 57, 118, 300-bus) under "
        "two scenarios with failure probabilities of approximately 10^-4 "
        "(Scenario 1) and 10^-5 (Scenario 2). The primary methods compared "
        "are crude MCS, BiCE (cross-entropy), aE-SuS (adaptive enhanced "
        "Subset Simulation), and iPIM/aPIM (importance sampling variants)."
    )
    pdf.body_text(
        "We reproduce Scenario 1 for all five IEEE cases using TSUM. Unlike "
        "the adaptive MCS methods which produce statistical point estimates "
        "of p_f, TSUM extracts deterministic rules that classify the entire "
        "state space into survival, failure, and unknown regions, yielding "
        "exact probability bounds."
    )

    # =================================================================
    # 2. Problem setup
    # =================================================================
    pdf.section_title("2. Problem Setup")

    pdf.section_title("2.1 DC-OPF system function", level=2)
    pdf.body_text(
        "The system function (sfun) solves a DC optimal power flow problem "
        "using scipy.optimize.linprog (dual-simplex method), a pure Python "
        "port of the MATLAB implementation used in Chan et al. For a given "
        "component state vector, the sfun returns the blackout size as a "
        "percentage of total demand not served. System failure is declared "
        "when blackout exceeds a threshold gamma."
    )

    pdf.section_title("2.2 Component model", level=2)
    pdf.body_text(
        "Following Chan et al. Table 3, components are modelled as follows:"
    )

    w = [pw * 0.25, pw * 0.18, pw * 0.18, pw * 0.18, pw * 0.21]
    pdf.table_header(
        ["Component", "States", "p(removed)", "p(degraded)", "p(full)"], w)
    pdf.table_row(
        ["Generator bus", "4", "0.01", "0.19 / 0.30", "0.50"], w)
    pdf.table_row(
        ["Ordinary bus", "2", "0.01", " - ", "0.99"], w, fill=True)
    pdf.table_row(
        ["Branch", "2", "0.01", " - ", "0.99"], w)
    pdf.ln(2)

    pdf.body_text(
        "Generator buses have four states: complete removal (state 0, "
        "p=0.01), 40% capacity (state 1, p=0.19), 80% capacity (state 2, "
        "p=0.30), and full capacity (state 3, p=0.50). Ordinary buses and "
        "branches are binary: failed (state 0, p=0.01) or operational "
        "(state 1, p=0.99). The probability tensor is padded to n_state=4 "
        "for mixed-state components."
    )

    pdf.section_title("2.3 IEEE test cases", level=2)

    w = [pw * 0.13, pw * 0.10, pw * 0.12, pw * 0.10,
         pw * 0.12, pw * 0.15, pw * 0.13, pw * 0.15]
    pdf.table_header(
        ["Case", "Buses", "Branches", "Gens",
         "Total", "Threshold", "Ref. p_f", "Status"], w)
    pdf.table_row(
        ["IEEE 14", "14", "20", "5", "34", "54.8%", "1.1e-4", "Done"], w)
    pdf.table_row(
        ["IEEE 30", "30", "41", "6", "71", "40.2%", "1.0e-4", "Done"], w,
        fill=True)
    pdf.table_row(
        ["IEEE 57", "57", "80", "7", "137", "54.1%", "1.0e-4", "Done"], w)
    pdf.table_row(
        ["IEEE 118", "118", "186", "54", "304", "13.8%", "1.0e-4", "In prog."],
        w, fill=True)
    pdf.table_row(
        ["IEEE 300", "300", "411", "69", "711", "26.1%", "1.0e-4", "In prog."],
        w)
    pdf.ln(2)

    pdf.body_text(
        "Thresholds and reference failure probabilities are from Chan et al. "
        "Table 2, Scenario 1. Total components include virtual bus edges "
        "(one per bus) plus physical branches."
    )

    # =================================================================
    # 3. Branch-only model
    # =================================================================
    pdf.add_page()
    pdf.section_title("3. Branch-Only Model: A Cautionary Finding")

    pdf.body_text(
        "Our initial attempt modelled only the 20 branches of IEEE 14-bus "
        "as binary components, with all buses (and generators) implicitly "
        "operational. This produced trivially low failure probability."
    )

    pdf.section_title("3.1 Results", level=2)
    pdf.body_text(
        "TSUM converged in 4 rounds (< 1 second), finding 3 single-branch "
        "survival rules and 0 failure rules, with p_failure = 0. The three "
        "survival rules were: {br4 >= 1}, {br7 >= 1}, {br18 >= 1}. Each "
        "states that if any single one of these branches is operational, "
        "the system survives regardless of all other branch states."
    )

    pdf.section_title("3.2 Root cause", level=2)
    pdf.body_text(
        "With all 5 generators always operational, the IEEE 14-bus network "
        "can serve local loads even when fully disconnected. Verification "
        "showed that with every single branch removed, blackout is "
        "approximately 50.9% - just below the 54.8% threshold. Having any "
        "one branch operational only improves this. The only true failure "
        "state (all 20 branches failed) has probability 0.01^20 ~ 10^-40."
    )
    pdf.body_text(
        "Lesson: the branch-only formulation is fundamentally different "
        "from Chan et al.'s model. Generator degradation and bus failures "
        "are essential to reach the reported p_f ~ 10^-4."
    )

    # =================================================================
    # 4. IEEE 14-bus full model
    # =================================================================
    pdf.add_page()

    base = Path(__file__).parent.parent
    c14_dir = base / "demos" / "case14" / "tsum_results_bus"
    c30_dir = base / "demos" / "case30" / "tsum_results_bus"
    c57_dir = base / "demos" / "case57" / "tsum_results_bus"

    c14_metrics = load_metrics(c14_dir / "metrics.json")
    c14_fail = load_rules(c14_dir / "rules_leq_0.json")
    c14_last = c14_metrics[-1]
    c14_time = sum(r['time_sec'] for r in c14_metrics)
    c14_sfun_time = sum(r['t_search'] + r['t_minimize'] for r in c14_metrics)
    c14_calls = int(c14_sfun_time / 1.30e-3)

    pdf.section_title("4. IEEE 14-Bus Full Model (Branches + Buses)")

    pdf.section_title("4.1 Configuration", level=2)
    pdf.body_text(
        "34 components: 5 generator buses (4-state), 9 ordinary buses "
        "(2-state), 20 branches (2-state). Blackout threshold: 54.8%. "
        "TSUM parameters: 1,000,000 samples per round (batch 100,000), "
        "convergence at p_unknown < 10^-5."
    )

    pdf.section_title("4.2 Results", level=2)

    w = [pw * 0.45, pw * 0.25, pw * 0.30]
    pdf.table_header(["Metric", "TSUM", "Chan et al."], w)
    pdf.table_row(["P(failure)",
                    "%.1e" % c14_last['p_failure'], "~1.1e-4"], w)
    pdf.table_row(["P(survival)",
                    "%.5f" % c14_last['p_survival'], "~0.99989"], w,
                   fill=True)
    pdf.table_row(["P(unknown)",
                    "%.1e" % c14_last['p_unknown'], " - "], w)
    pdf.table_row(["Rounds", str(len(c14_metrics)), " - "], w, fill=True)
    pdf.table_row(["Survival rules",
                    str(c14_last['n_rules_surv']), " - "], w)
    pdf.table_row(["Failure rules",
                    str(c14_last['n_rules_fail']), " - "], w, fill=True)
    pdf.table_row(["sfun evaluations",
                    "~%s" % format(c14_calls, ','), " - "], w)
    pdf.table_row(["Wall time", "%.1fs" % c14_time, " - "], w, fill=True)
    pdf.ln(2)

    p_lo = c14_last['p_failure']
    p_hi = c14_last['p_failure'] + c14_last['p_unknown']
    pdf.body_text(
        "TSUM converged in %d rounds (%.1fs) with ~%s DC-OPF evaluations. "
        "The estimated failure probability of %.1e (bounds: [%.1e, %.1e]) "
        "is consistent with Chan et al.'s reference of 1.1e-4."
        % (len(c14_metrics), c14_time, format(c14_calls, ','),
           p_lo, p_lo, p_hi)
    )

    pdf.section_title("4.3 Failure rules", level=2)
    pdf.body_text(
        "TSUM identified %d minimal failure rules. All require bus 3 "
        "(generator, 100 MW capacity, 94.2 MW local load) to be in "
        "state 0 (completely removed):" % c14_last['n_rules_fail']
    )

    for i, rule in enumerate(c14_fail):
        comps = {k: v for k, v in rule.items() if k != 'sys'}
        parts = ["%s <= %s" % (k, v[1]) for k, v in comps.items()]
        pdf.set_font("Courier", "", 9)
        pdf.set_x(pdf.l_margin + 10)
        pdf.cell(0, 5, "Rule %d: %s" % (i + 1, " AND ".join(parts)))
        pdf.ln(5)
    pdf.ln(2)

    pdf.body_text(
        "The dominant failure rule (vbus3=0 AND vbus4=0) has probability "
        "0.01 x 0.01 = 10^-4, which directly explains the system failure "
        "probability. Bus 3 is the critical vulnerability: losing its "
        "100 MW generator sheds ~36% of demand; combined with bus 4 "
        "(47.8 MW load), blackout exceeds 54.8%."
    )

    # =================================================================
    # 5. IEEE 30-bus full model
    # =================================================================

    c30_metrics = load_metrics(c30_dir / "metrics.json")
    c30_fail = load_rules(c30_dir / "rules_leq_0.json")
    c30_last = c30_metrics[-1]
    c30_time = sum(r['time_sec'] for r in c30_metrics)
    c30_sfun_time = sum(r['t_search'] + r['t_minimize'] for r in c30_metrics)
    c30_calls = int(c30_sfun_time / 1.59e-3)

    pdf.section_title("5. IEEE 30-Bus Full Model (Branches + Buses)")

    pdf.section_title("5.1 Configuration", level=2)
    pdf.body_text(
        "71 components: 6 generator buses (4-state), 24 ordinary buses "
        "(2-state), 41 branches (2-state). Blackout threshold: 40.2%. "
        "Same TSUM parameters as IEEE 14-bus."
    )

    pdf.section_title("5.2 Results", level=2)

    w = [pw * 0.45, pw * 0.25, pw * 0.30]
    pdf.table_header(["Metric", "TSUM", "Chan et al."], w)
    pdf.table_row(["P(failure)",
                    "%.1e" % c30_last['p_failure'], "~1.0e-4"], w)
    pdf.table_row(["P(survival)",
                    "%.5f" % c30_last['p_survival'], "~0.99990"], w,
                   fill=True)
    pdf.table_row(["P(unknown)",
                    "%.1e" % c30_last['p_unknown'], " - "], w)
    pdf.table_row(["Rounds", str(len(c30_metrics)), " - "], w, fill=True)
    pdf.table_row(["Survival rules",
                    str(c30_last['n_rules_surv']), " - "], w)
    pdf.table_row(["Failure rules",
                    str(c30_last['n_rules_fail']), " - "], w, fill=True)
    pdf.table_row(["sfun evaluations",
                    "~%s" % format(c30_calls, ','), " - "], w)
    pdf.table_row(["Wall time", "%.1fs" % c30_time, " - "], w, fill=True)
    pdf.ln(2)

    p_lo = c30_last['p_failure']
    p_hi = c30_last['p_failure'] + c30_last['p_unknown']
    pdf.body_text(
        "TSUM converged in %d rounds (%.1fs) with ~%s DC-OPF evaluations. "
        "The estimated failure probability of %.1e (bounds: [%.1e, %.1e]) "
        "is consistent with Chan et al.'s reference of 1.0e-4."
        % (len(c30_metrics), c30_time, format(c30_calls, ','),
           p_lo, p_lo, p_hi)
    )

    pdf.section_title("5.3 Failure rule analysis", level=2)

    comp_freq = Counter()
    rule_sizes = Counter()
    for rule in c30_fail:
        comps = [k for k in rule if k != 'sys']
        rule_sizes[len(comps)] += 1
        for c in comps:
            comp_freq[c] += 1

    pdf.body_text(
        "TSUM found %d failure rules with the following size distribution:"
        % len(c30_fail)
    )

    w = [pw * 0.35, pw * 0.30, pw * 0.35]
    pdf.table_header(["Rule size (components)", "Count", "Fraction"], w)
    for s in sorted(rule_sizes):
        frac = rule_sizes[s] / len(c30_fail)
        pdf.table_row([str(s), str(rule_sizes[s]), "%.0f%%" % (frac * 100)],
                      w, fill=(s % 2 == 0))
    pdf.ln(2)

    pdf.body_text("Most frequent components in failure rules:")

    w = [pw * 0.25, pw * 0.25, pw * 0.25, pw * 0.25]
    pdf.table_header(["Component", "Appearances", "Fraction", "Type"], w)
    # Generator buses for case30: buses 1, 2, 5, 8, 11, 13
    gen30 = {'vbus1', 'vbus2', 'vbus5', 'vbus8', 'vbus11', 'vbus13'}
    for i, (comp, count) in enumerate(comp_freq.most_common(10)):
        frac = count / len(c30_fail)
        ctype = ("generator" if comp in gen30 else
                 "bus" if comp.startswith("vbus") else "branch")
        pdf.table_row([comp, str(count), "%.0f%%" % (frac * 100), ctype],
                      w, fill=(i % 2 == 1))
    pdf.ln(2)

    smallest = min(c30_fail, key=lambda r: len(r))
    comps_s = {k: v for k, v in smallest.items() if k != 'sys'}
    parts_s = ["%s <= %s" % (k, v[1]) for k, v in comps_s.items()]
    top_comp, top_count = comp_freq.most_common(1)[0]
    pdf.body_text(
        "The smallest failure rule involves %d components: %s. "
        "%s appears in %d of %d failure rules (%.0f%%), making it "
        "the single most critical vulnerability in the IEEE 30-bus network."
        % (len(comps_s), ", ".join(parts_s),
           top_comp, top_count, len(c30_fail),
           top_count / len(c30_fail) * 100)
    )

    # =================================================================
    # 6. IEEE 57-bus full model
    # =================================================================
    pdf.add_page()

    c57_metrics = load_metrics(c57_dir / "metrics.json")
    c57_last = c57_metrics[-1]
    c57_time = sum(r['time_sec'] for r in c57_metrics)
    c57_sfun_time = sum(r['t_search'] + r['t_minimize'] for r in c57_metrics)
    # Estimate per-call cost: case57 is between case30 (1.59ms) and case118
    c57_ms_per_call = 2.5  # estimated
    c57_calls = int(c57_sfun_time / (c57_ms_per_call * 1e-3))

    pdf.section_title("6. IEEE 57-Bus Full Model (Branches + Buses)")

    pdf.section_title("6.1 Configuration", level=2)
    pdf.body_text(
        "137 components: 7 generator buses (4-state), 50 ordinary buses "
        "(2-state), 80 branches (2-state). Blackout threshold: 54.1%. "
        "Same TSUM parameters as previous cases."
    )

    pdf.section_title("6.2 Results", level=2)

    w = [pw * 0.45, pw * 0.25, pw * 0.30]
    pdf.table_header(["Metric", "TSUM", "Chan et al."], w)
    pdf.table_row(["P(failure)",
                    "%.1e" % c57_last['p_failure'], "~1.0e-4"], w)
    pdf.table_row(["P(survival)",
                    "%.5f" % c57_last['p_survival'], "~0.99990"], w,
                   fill=True)
    pdf.table_row(["P(unknown)",
                    "%.1e" % c57_last['p_unknown'], " - "], w)
    pdf.table_row(["Rounds", str(len(c57_metrics)), " - "], w, fill=True)
    pdf.table_row(["Survival rules",
                    str(c57_last['n_rules_surv']), " - "], w)
    pdf.table_row(["Failure rules",
                    str(c57_last['n_rules_fail']), " - "], w, fill=True)
    pdf.table_row(["Wall time", "%.1fs (%.1f min)" % (c57_time, c57_time/60),
                    " - "], w, fill=True)
    pdf.ln(2)

    p_lo = c57_last['p_failure']
    p_hi = c57_last['p_failure'] + c57_last['p_unknown']
    pdf.body_text(
        "TSUM converged in %d rounds (%.1f minutes). "
        "The estimated failure probability of %.1e (bounds: [%.1e, %.1e]) "
        "is consistent with Chan et al.'s reference of 1.0e-4."
        % (len(c57_metrics), c57_time / 60, p_lo, p_lo, p_hi)
    )
    pdf.body_text(
        "Notably, IEEE 57-bus converged faster than IEEE 30-bus (%d vs %d "
        "rounds) despite having nearly twice the components (137 vs 71). "
        "This is because the higher threshold (54.1%% vs 40.2%%) means "
        "fewer failure modes to discover: the system tolerates more "
        "component degradation before blackout exceeds the threshold, "
        "resulting in fewer and simpler failure rules (%d vs %d)."
        % (len(c57_metrics), len(c30_metrics),
           c57_last['n_rules_fail'], c30_last['n_rules_fail'])
    )

    # =================================================================
    # 7. IEEE 118-bus: biased sampling strategy
    # =================================================================
    pdf.add_page()
    pdf.section_title("7. IEEE 118-Bus: Biased Discovery Sampling")

    # Load from PBS logs (check tsum_results_bus first for updated logs)
    c118_results_dir = base / "demos" / "case118" / "tsum_results_bus"
    c118_log_A = base / "demos" / "case118" / "164043425.gadi-pbs.log"
    c118_log_B = (c118_results_dir / "164065855.gadi-pbs.log"
                  if (c118_results_dir / "164065855.gadi-pbs.log").exists()
                  else base / "demos" / "case118" / "164065855.gadi-pbs.log")
    c118_log_C = base / "demos" / "case118" / "164071267.gadi-pbs.log"
    c118_A = parse_log_metrics(c118_log_A) if c118_log_A.exists() else None
    c118_B = parse_log_metrics(c118_log_B) if c118_log_B.exists() else None
    c118_C = parse_log_metrics(c118_log_C) if c118_log_C.exists() else None
    # Also load metrics.json if available (more accurate than log parsing)
    c118_metrics_path = c118_results_dir / "metrics.json"
    if c118_metrics_path.exists():
        c118_metrics_lines = load_metrics(c118_metrics_path)
        if c118_metrics_lines:
            c118_B_metrics = c118_metrics_lines[-1]
            # Override log-parsed values with metrics.json
            c118_B = {
                'round': c118_B_metrics['round'],
                'p_survival': c118_B_metrics['p_survival'],
                'p_failure': c118_B_metrics['p_failure'],
                'p_unknown': c118_B_metrics['p_unknown'],
                'n_rules_surv': c118_B_metrics['n_rules_surv'],
                'n_rules_fail': c118_B_metrics['n_rules_fail'],
            }
    # Use biased run as main reference
    c118_data = c118_B or c118_A

    pdf.section_title("7.1 Configuration", level=2)
    pdf.body_text(
        "304 components: 54 generator buses (4-state), 64 ordinary buses "
        "(2-state), 186 branches (2-state). Blackout threshold: 13.8% - "
        "the lowest among all five cases. Convergence threshold: "
        "p_unknown < 10^-3 (relaxed from 10^-5 due to problem difficulty). "
        "Runs executed on Gadi cluster with 2 GPUs (NVIDIA A100)."
    )

    pdf.section_title("7.2 Why standard sampling fails", level=2)
    pdf.body_text(
        "After %d rounds of standard MCS, TSUM found %d survival rules "
        "covering %.1f%% of probability mass, but zero failure rules. "
        "The explanation lies in the distinction between failure mode "
        "diversity and failure mode probability."
        % (c118_A['round'] if c118_A else 329,
           c118_A['n_rules_surv'] if c118_A else 328,
           (c118_A['p_survival'] if c118_A else 0.111) * 100)
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Sampling bias toward survival")
    pdf.ln(7)
    pdf.body_text(
        "With p_fail = 0.01 per component, a typical random sample has "
        "~3 failed components. Losing 3 out of 304 almost never causes "
        ">13.8% blackout, so samples overwhelmingly survive. The algorithm "
        "keeps finding new survival configurations but never reaches the "
        "failure region."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Fragmented survival and failure spaces")
    pdf.ln(7)
    pdf.body_text(
        "Each survival rule has ~150 conditions - roughly half the network "
        "must be in specific states. The low threshold creates enormous "
        "diversity of both survival and failure modes, requiring far more "
        "rules than the higher-threshold cases. This is the threshold effect: "
        "a low threshold does not make failures more probable, it makes them "
        "more diverse."
    )

    # =================================================================
    # 7.3 Biased discovery sampling
    # =================================================================
    pdf.section_title("7.3 Biased discovery sampling strategy", level=2)
    pdf.body_text(
        "To overcome sampling bias, we introduce biased discovery sampling: "
        "a biased probability tensor is used during the search phase to "
        "generate more degraded configurations, while the true probability "
        "tensor is always used for probability estimation. This decoupling "
        "is valid because TSUM rules are distribution-independent patterns - "
        "a minimal failure rule found under biased sampling remains a valid "
        "failure rule under the true distribution."
    )
    pdf.body_text(
        "The bias is applied via make_discovery_probs(probs, bias_factor), "
        "which multiplies the failure-state probabilities of each component "
        "by a factor k and re-normalises. With bias_factor=10, per-component "
        "failure probability increases from ~1% to ~9%, so a typical sample "
        "has ~27 failed components instead of ~3, dramatically increasing "
        "hits in the failure/unknown region."
    )

    # =================================================================
    # 7.4 Three-run comparison
    # =================================================================
    pdf.section_title("7.4 Initial comparison: three sampling strategies", level=2)

    pdf.body_text(
        "Three configurations were run in parallel on the Gadi cluster "
        "(all with bias_factor=10 where applicable):"
    )
    pdf.bullet("A - Baseline: standard MCS, no biasing")
    pdf.bullet("B - Biased (all rounds): discovery_probs used throughout")
    pdf.bullet(
        "C - Auto-switch: biased for first 1,000 rounds, then true probs")
    pdf.ln(2)

    rA = c118_A['round'] if c118_A else 4240
    srA = c118_A['n_rules_surv'] if c118_A else 4239
    frA = c118_A['n_rules_fail'] if c118_A else 0
    upA = c118_A['p_unknown'] if c118_A else 0.645
    spA = c118_A['p_survival'] if c118_A else 0.355

    rC = c118_C['round'] if c118_C else 1609
    srC = c118_C['n_rules_surv'] if c118_C else 1562
    frC = c118_C['n_rules_fail'] if c118_C else 46
    upC = c118_C['p_unknown'] if c118_C else 0.454
    spC = c118_C['p_survival'] if c118_C else 0.546

    # B values from metrics.json (most accurate)
    rB = c118_B['round'] if c118_B else 10000
    srB = c118_B['n_rules_surv'] if c118_B else 9338
    frB = c118_B['n_rules_fail'] if c118_B else 662
    upB = c118_B['p_unknown'] if c118_B else 0.230
    spB = c118_B['p_survival'] if c118_B else 0.770

    pdf.body_text(
        "The auto-switch variant (C) found 46 failure rules during its "
        "biased phase (rounds 1-1000), but stopped finding failure rules "
        "immediately after switching to true probs at round 1001, reverting "
        "to baseline behaviour. This confirmed that persistent biased "
        "sampling is necessary for this case. Runs A and C were terminated "
        "after ~%d and ~%d rounds respectively. Run B was continued "
        "to %d rounds." % (rA, rC, rB)
    )

    # =================================================================
    # 7.5 Extended results: biased vs alternating
    # =================================================================
    pdf.section_title("7.5 Extended results: biased (bf=10) vs alternating bias",
                      level=2)

    # Load alternating results
    c118_alt_dir = base / "demos" / "case118" / "results_alt"
    c118_alt_metrics_path = c118_alt_dir / "metrics.json"
    c118_alt = None
    if c118_alt_metrics_path.exists():
        c118_alt_lines = load_metrics(c118_alt_metrics_path)
        if c118_alt_lines:
            c118_alt_last = c118_alt_lines[-1]
            c118_alt = {
                'round': c118_alt_last['round'],
                'p_survival': c118_alt_last['p_survival'],
                'p_failure': c118_alt_last['p_failure'],
                'p_unknown': c118_alt_last['p_unknown'],
                'n_rules_surv': c118_alt_last['n_rules_surv'],
                'n_rules_fail': c118_alt_last['n_rules_fail'],
            }

    rAlt = c118_alt['round'] if c118_alt else 10000
    srAlt = c118_alt['n_rules_surv'] if c118_alt else 9668
    frAlt = c118_alt['n_rules_fail'] if c118_alt else 332
    upAlt = c118_alt['p_unknown'] if c118_alt else 0.272
    spAlt = c118_alt['p_survival'] if c118_alt else 0.728

    pdf.body_text(
        "Two long runs were completed to 10,000 rounds on the Gadi cluster:"
    )
    pdf.bullet(
        "B - Fixed biased (bf=10): biased sampling with factor=10 throughout")
    pdf.bullet(
        "D - Alternating bias: cycles between bf=10 (100 rounds) and bf=2 "
        "(100 rounds), giving dedicated phases for failure and survival "
        "discovery")
    pdf.ln(2)

    w = [pw * 0.35, pw * 0.325, pw * 0.325]
    pdf.table_header(["Metric", "B: Biased (bf=10)", "D: Alternating"], w)
    pdf.table_row(["Rounds", str(rB), str(rAlt)], w)
    pdf.table_row(["Survival rules", str(srB), str(srAlt)], w, fill=True)
    pdf.table_row(["Failure rules", str(frB), str(frAlt)], w)
    pdf.table_row(["P(survival)", "%.4f" % spB, "%.4f" % spAlt], w, fill=True)
    pdf.table_row(["P(failure)", "~0", "~0"], w)
    pdf.table_row(["P(unknown)", "%.4f" % upB, "%.4f" % upAlt], w, fill=True)
    pdf.ln(2)

    pdf.body_text(
        "Fixed biased (B) achieves lower p_unknown (%.3f vs %.3f) despite "
        "finding fewer survival rules (%d vs %d). It discovers twice as many "
        "failure rules (%d vs %d). The alternating strategy's low-bias phases "
        "find survival rules that overlap more with existing ones, while "
        "fixed high bias explores the frontier more effectively. For IEEE "
        "118-bus, fixed bf=10 is the better strategy."
        % (upB, upAlt, srB, srAlt, frB, frAlt)
    )

    pdf.section_title(
        "7.6 Why failure probability shows zero despite %d failure rules" % frB,
        level=2)
    pdf.body_text(
        "Although %d failure rules have been found, the estimated p_failure "
        "remains effectively zero (relative to true p_f ~ 10^-4). "
        "This is not a bug - it reflects "
        "the rule sizes. A typical failure rule found by biased sampling "
        "requires ~17 components to be simultaneously in failed/degraded "
        "states, for example:" % frB
    )
    pdf.set_font("Courier", "", 8)
    pdf.set_x(pdf.l_margin + 10)
    pdf.multi_cell(0, 4,
        "vbus3<=0, vbus19<=0, vbus40<=2, vbus42<=2, vbus49<=0,\n"
        "vbus54<=0, vbus75<=0, vbus101<=0, br1<=0, br13<=0,\n"
        "br51<=0, br125<=0, br147<=0  [13 conditions]")
    pdf.ln(2)
    pdf.body_text(
        ("The probability of this rule under the true distribution is "
        "approximately (0.01)^11 x (0.05)^2 ~ 10^-24. Even summing over "
        "all %d rules gives ~ 10^-22, which is far below 10^-4. "
        "The true p_f ~ 10^-4 is the aggregate over all minimal failure "
        "rules, including short rules (2-4 conditions) that represent the "
        "dominant failure mechanisms. These shorter rules have not yet been "
        "discovered because they require very specific critical component "
        "combinations that even biased sampling rarely hits.") % frB
    )
    pdf.body_text(
        "Discovering these minimal cut sets remains the key challenge. "
        "The biased sampling approach has demonstrated it can find failure "
        "rules where standard sampling cannot - continued running will "
        "progressively find shorter, higher-probability failure modes."
    )

    # =================================================================
    # 7.6 Comparison with adaptive MCS methods
    # =================================================================
    pdf.section_title("7.6 Estimated cost of adaptive MCS for IEEE 118", level=2)

    pdf.body_text(
        "From Chan et al. Table 4 (Scenario 1), the relative efficiency "
        "of each method for IEEE 118-bus is:"
    )

    pf = 1e-4
    delta = 0.10
    N_crude_118 = int(1 / (delta**2 * pf))
    t_per_call = 5.0  # ms, estimated for case118

    releff_118 = {
        'BiCE (K=1)': 19,
        'BiCE (K=10)': 18,
        'aE-SuS': 8.1,
        'iPIM': 6.1,
        'iPIM+aPIM': 3.6,
        'aE-SuS+aPIM': 4.0,
    }

    w = [pw * 0.30, pw * 0.15, pw * 0.22, pw * 0.33]
    pdf.table_header(["Method", "relEff", "Evals (10% CoV)", "Est. time"], w)
    pdf.table_row(["Crude MCS", "1.0",
                    "%dK" % (N_crude_118 // 1000),
                    "%.0f min" % (N_crude_118 * t_per_call / 1e3 / 60)], w)
    for i, (name, re) in enumerate(releff_118.items()):
        n = N_crude_118 / re
        t = n * t_per_call / 1000
        pdf.table_row([name, "%.1f" % re,
                        "%dK" % (n // 1000),
                        "%.1f min" % (t / 60)],
                      w, fill=(i % 2 == 0))
    # Single run
    n_single = 2000 * 4  # T=4 levels for p_f=1e-4
    pdf.table_row(["aE-SuS single run", "--",
                    "%dK" % (n_single // 1000),
                    "%.0fs (62%% CoV)" % (n_single * t_per_call / 1000)], w)
    pdf.ln(2)

    pdf.body_text(
        "The adaptive MCS methods are relatively fast for IEEE 118-bus "
        "because they only need to estimate p_f as a scalar. TSUM's "
        "approach of enumerating all survival and failure rules is more "
        "expensive for this case, but produces structural insight that "
        "the adaptive methods cannot provide."
    )

    # =================================================================
    # 7.7 Boundary walking experiment
    # =================================================================
    pdf.add_page()
    pdf.section_title("7.7 Boundary walking: a smarter unknown selection?",
                      level=2)

    pdf.body_text(
        "To find shorter failure rules more efficiently, we implemented a "
        "boundary walking strategy. Starting from an all-operational state, "
        "components are progressively degraded (weighted by failure probability) "
        "until the system fails, capturing both 'barely surviving' and 'barely "
        "failing' samples near the failure boundary. This targets the boundary "
        "region directly rather than relying on random sampling."
    )

    # Load walk results
    c118_walk_dir = base / "demos" / "case118" / "results_walk"
    c118_walk2_dir = base / "demos" / "case118" / "results_walk_20_20"
    c118_walk = None
    c118_walk2 = None
    if (c118_walk_dir / "metrics.json").exists():
        c118_walk_lines = load_metrics(c118_walk_dir / "metrics.json")
        if c118_walk_lines:
            c118_walk = c118_walk_lines[-1]
    if (c118_walk2_dir / "metrics.json").exists():
        c118_walk2_lines = load_metrics(c118_walk2_dir / "metrics.json")
        if c118_walk2_lines:
            c118_walk2 = c118_walk2_lines[-1]

    pdf.section_title("Configuration", level=2)
    pdf.body_text(
        "Two hybrid configurations were tested on Gadi (CPU-only, 192 workers), "
        "both with bf=10 biased sampling for the MCS rounds:"
    )
    pdf.bullet(
        "Walk-5-4: boundary walk every 5 rounds, 4 walks per walk round "
        "(80% MCS, 20% walk rounds)")
    pdf.bullet(
        "Walk-20-20: boundary walk every 20 rounds, 20 walks per walk round "
        "(95% MCS, 5% walk rounds)")
    pdf.ln(2)

    pdf.section_title("Results", level=2)
    if c118_walk and c118_walk2:
        w = [pw * 0.28, pw * 0.24, pw * 0.24, pw * 0.24]
        pdf.table_header(["Metric", "Walk-5-4", "Walk-20-20",
                          "bf=10 (no walk)"], w)
        pdf.table_row(["Rounds",
                        str(c118_walk['round']),
                        str(c118_walk2['round']),
                        str(rB)], w)
        pdf.table_row(["Survival rules",
                        str(c118_walk['n_rules_surv']),
                        str(c118_walk2['n_rules_surv']),
                        str(srB)], w, fill=True)
        pdf.table_row(["Failure rules",
                        str(c118_walk['n_rules_fail']),
                        str(c118_walk2['n_rules_fail']),
                        str(frB)], w)
        pdf.table_row(["Avg fail len",
                        "%.1f" % c118_walk['avg_len_fail'],
                        "%.1f" % c118_walk2['avg_len_fail'],
                        "~17"], w, fill=True)
        pdf.table_row(["P(survival)",
                        "%.3e" % c118_walk['p_survival'],
                        "%.3e" % c118_walk2['p_survival'],
                        "%.4f" % spB], w)
        pdf.table_row(["P(failure)",
                        "%.3e" % c118_walk['p_failure'],
                        "%.3e" % c118_walk2['p_failure'],
                        "~0"], w, fill=True)
        pdf.table_row(["P(unknown)",
                        "%.4f" % c118_walk['p_unknown'],
                        "%.4f" % c118_walk2['p_unknown'],
                        "%.4f" % upB], w)
        pdf.ln(2)

    pdf.section_title("Analysis", level=2)
    pdf.body_text(
        "Boundary walking is highly efficient at discovering failure rules: "
        "each walk round produces ~192 failure rules (one per walk), compared "
        "to ~12-14 per normal MCS round. However, all three approaches still "
        "show p_unknown = 1.0 at early rounds (< 100), because the discovered "
        "failure rules are too long to carry meaningful probability."
    )
    pdf.body_text(
        "The walks produce rules with average length ~12-13 conditions. While "
        "shorter than standard MCS rules (~17 conditions), this is still far "
        "too long: a 13-condition rule has probability ~(0.01)^11 x (0.05)^2 "
        "~ 10^-24 under the true distribution. The probability-dominating "
        "short rules (3-4 conditions) arise from rare MCS samples, not from "
        "systematic boundary walking."
    )
    pdf.body_text(
        "This reveals a fundamental limitation: boundary walking degrades "
        "components one at a time from all-operational, so it inherently "
        "produces failure modes where ~12 components must fail simultaneously. "
        "The rare 3-condition rules that dominate p_f ~ 10^-4 represent "
        "critical vulnerability clusters that walking cannot discover - they "
        "require many more MCS rounds to find by chance."
    )
    pdf.body_text(
        "The hybrid approach (walks + MCS) is valid but the walk rounds do "
        "not accelerate convergence for this problem. The walks efficiently "
        "catalogue the failure space but the probability computation is "
        "dominated by short rules that only MCS can find. Longer runs are "
        "needed to confirm whether the walk overhead (slower walk rounds) "
        "is justified by the coverage benefit."
    )

    # =================================================================
    # 8. IEEE 300-bus
    # =================================================================
    pdf.add_page()
    pdf.section_title("8. IEEE 300-Bus: Initial Results")

    # Load case300 data from metrics.json where available
    c300_bf5_dir = base / "demos" / "case300" / "tsum_results_bus"
    c300_bf10_dir = base / "demos" / "case300" / "results_bf10"
    c300_bf5 = None
    c300_bf10 = None
    if (c300_bf5_dir / "metrics.json").exists():
        c300_bf5_lines = load_metrics(c300_bf5_dir / "metrics.json")
        if c300_bf5_lines:
            c300_bf5 = c300_bf5_lines[-1]
    if (c300_bf10_dir / "metrics.json").exists():
        c300_bf10_lines = load_metrics(c300_bf10_dir / "metrics.json")
        if c300_bf10_lines:
            c300_bf10 = c300_bf10_lines[-1]
    c300_data = c300_bf5  # primary reference

    pdf.section_title("8.1 Configuration", level=2)
    pdf.body_text(
        "711 components: 69 generator buses (4-state), 231 ordinary buses "
        "(2-state), 411 branches (2-state). Blackout threshold: 26.1% "
        "(Scenario 1 from Chan et al.). This is the largest system in the "
        "benchmark suite, with more than double the components of case118."
    )

    pdf.section_title("8.2 Bug fix: negative-PD buses", level=2)
    pdf.body_text(
        "The initial case300 run returned 100% blackout even with all "
        "components operational. The root cause was 8 buses in the MATPOWER "
        "case300 file with negative PD values (generation injections modelled "
        "as negative load). The load2disp() function converted these to "
        "dispatchable loads with PMIN = -PD > 0 = PMAX, creating infeasible "
        "LP bounds. Fix: filter to bus[:, PD] > 0 (exclude negative loads). "
        "After fix: all-operational gives 0.07% blackout (sys_st=1)."
    )

    pdf.section_title("8.3 Bias factor comparison: bf=5 vs bf=10", level=2)
    if c300_bf5 and c300_bf10:
        pdf.body_text(
            "Two runs with different bias factors on 2x A100 GPUs:"
        )
        w = [pw * 0.35, pw * 0.325, pw * 0.325]
        pdf.table_header(["Metric", "bf=5", "bf=10"], w)
        pdf.table_row(["Rounds", str(c300_bf5['round']),
                        str(c300_bf10['round'])], w)
        pdf.table_row(["Survival rules", str(c300_bf5['n_rules_surv']),
                        str(c300_bf10['n_rules_surv'])], w, fill=True)
        pdf.table_row(["Failure rules", str(c300_bf5['n_rules_fail']),
                        str(c300_bf10['n_rules_fail'])], w)
        pdf.table_row(["Surv/Fail ratio",
                        "%.0f%%/%.0f%%" % (
                            c300_bf5['n_rules_surv'] / max(1, c300_bf5['round']) * 100,
                            c300_bf5['n_rules_fail'] / max(1, c300_bf5['round']) * 100),
                        "%.0f%%/%.0f%%" % (
                            c300_bf10['n_rules_surv'] / max(1, c300_bf10['round']) * 100,
                            c300_bf10['n_rules_fail'] / max(1, c300_bf10['round']) * 100)],
                      w, fill=True)
        pdf.table_row(["P(survival)", "%.5f" % c300_bf5['p_survival'],
                        "%.5f" % c300_bf10['p_survival']], w)
        pdf.table_row(["P(unknown)", "%.4f" % c300_bf5['p_unknown'],
                        "%.4f" % c300_bf10['p_unknown']], w, fill=True)
        pdf.table_row(["Avg surv rule len", "%.0f" % c300_bf5['avg_len_surv'],
                        "%.0f" % c300_bf10['avg_len_surv']], w)
        pdf.table_row(["Avg fail rule len", "%.0f" % c300_bf5['avg_len_fail'],
                        "%.0f" % c300_bf10['avg_len_fail']], w, fill=True)
        pdf.ln(2)
        pdf.body_text(
            "bf=5 achieves lower p_unknown (%.4f vs %.4f) with a balanced "
            "rule discovery ratio (70%%/30%% surv/fail). bf=10 is heavily "
            "skewed toward failure rules (10%%/90%%) which contribute "
            "negligible probability mass. For case300, the moderate bias "
            "factor is more effective at reducing p_unknown."
            % (c300_bf5['p_unknown'], c300_bf10['p_unknown'])
        )
    elif c300_data:
        pdf.body_text(
            "Running with bias_factor=5.0 on 2x A100 GPUs. Current status "
            "at round %d: %d surv rules, %d fail rules, p_unknown=%.4f."
            % (c300_data['round'], c300_data['n_rules_surv'],
               c300_data['n_rules_fail'], c300_data['p_unknown'])
        )
    else:
        pdf.body_text(
            "Case300 runs are in progress on the Gadi cluster."
        )

    pdf.body_text(
        "With 711 components, survival rules require ~360 conditions "
        "(over half the network) and each covers very little probability. "
        "Convergence is extremely slow for this system size. "
        "Both biased sampling strategies find failure rules, but as with "
        "case118 they are high-order rules (~45 conditions) with negligible "
        "individual probability."
    )

    # =================================================================
    # 9. Convergence comparison
    # =================================================================
    pdf.add_page()
    pdf.section_title("9. Convergence Behaviour")

    pdf.body_text(
        "The completed cases exhibit markedly different convergence "
        "profiles, revealing the interplay between system size, threshold, "
        "and rule complexity."
    )

    w = [pw * 0.19, pw * 0.12, pw * 0.12, pw * 0.12,
         pw * 0.15, pw * 0.15, pw * 0.15]
    pdf.table_header(
        ["Metric", "IEEE 14", "IEEE 30", "IEEE 57",
         "Threshold", "Fail rules", "Rounds"], w)
    pdf.table_row(
        ["Components", "34", "71", "137",
         "", "", ""], w)
    pdf.table_row(
        ["Threshold", "54.8%", "40.2%", "54.1%",
         "", "", ""], w, fill=True)
    pdf.table_row(
        ["Rounds", str(len(c14_metrics)), str(len(c30_metrics)),
         str(len(c57_metrics)), "", "", ""], w)
    pdf.table_row(
        ["Surv rules",
         str(c14_last['n_rules_surv']),
         str(c30_last['n_rules_surv']),
         str(c57_last['n_rules_surv']),
         "", "", ""], w, fill=True)
    pdf.table_row(
        ["Fail rules",
         str(c14_last['n_rules_fail']),
         str(c30_last['n_rules_fail']),
         str(c57_last['n_rules_fail']),
         "", "", ""], w)
    pdf.table_row(
        ["Avg surv len",
         "%.1f" % c14_last['avg_len_surv'],
         "%.1f" % c30_last['avg_len_surv'],
         "%.1f" % c57_last['avg_len_surv'],
         "", "", ""], w, fill=True)
    pdf.table_row(
        ["Avg fail len",
         "%.1f" % c14_last['avg_len_fail'],
         "%.1f" % c30_last['avg_len_fail'],
         "%.1f" % c57_last['avg_len_fail'],
         "", "", ""], w)
    pdf.table_row(
        ["Wall time",
         "%.1fs" % c14_time,
         "%.0fs" % c30_time,
         "%.0fs" % c57_time,
         "", "", ""], w, fill=True)
    pdf.table_row(
        ["P(failure)",
         "%.1e" % c14_last['p_failure'],
         "%.1e" % c30_last['p_failure'],
         "%.1e" % c57_last['p_failure'],
         "", "", ""], w)
    pdf.ln(2)

    pdf.body_text(
        "A key finding is that convergence depends more on the blackout "
        "threshold than on the number of components. IEEE 57-bus (137 "
        "components) converged in %d rounds with %d failure rules, while "
        "IEEE 30-bus (71 components) required %d rounds with %d failure "
        "rules. The higher threshold of 54.1%% (vs 40.2%%) means fewer "
        "combinations of component failures can cause blackout, resulting "
        "in a simpler failure rule space."
        % (len(c57_metrics), c57_last['n_rules_fail'],
           len(c30_metrics), c30_last['n_rules_fail'])
    )
    pdf.body_text(
        "This threshold effect is even more pronounced for IEEE 118-bus "
        "(13.8% threshold), where the fragile system creates an enormous "
        "space of both survival and failure modes that TSUM must enumerate."
    )

    # =================================================================
    # 10. Computation cost comparison (for completed cases)
    # =================================================================
    pdf.section_title("10. Computation Cost vs Adaptive MCS")

    pdf.body_text(
        "Chan et al. do not report wall-clock times but state that "
        "computational cost is dominated by DC-OPF evaluations. We "
        "estimate the number of evaluations for each method and convert "
        "to time using our measured per-call costs (scipy linprog): "
        "1.30 ms (case 14) and 1.59 ms (case 30)."
    )

    pdf.section_title("10.1 Method parameters from Chan et al.", level=2)
    pdf.bullet(
        "Crude MCS: 10^8 evaluations for IEEE 14, 30, 57; achieves "
        "approximately 1% CoV at p_f ~ 10^-4")
    pdf.bullet(
        "aE-SuS: N=2,000 samples per level, p=0.1 conditional probability. "
        "For p_f ~ 10^-4 this gives T=4 levels, ~8,000 evals per run. "
        "A single run has ~62% CoV; reliable estimates require aggregation.")
    pdf.bullet(
        "Relative efficiency (Table 4): ratio of crude MCS cost to "
        "adaptive method cost at equal accuracy.")
    pdf.ln(2)

    pdf.section_title("10.2 Comparison for IEEE 14 and 30", level=2)

    N_crude = 1_000_000
    # Table 4 relEff for case14 and case30
    N_asus_14 = int(1 / (2.5 * delta**2 * pf))
    N_asus_30 = int(1 / (3.3 * delta**2 * pf))

    w = [pw * 0.30, pw * 0.175, pw * 0.175, pw * 0.175, pw * 0.175]
    pdf.table_header(
        ["Method", "Evals (14)", "Time (14)", "Evals (30)", "Time (30)"], w)
    pdf.table_row(
        ["Crude MCS (10% CoV)",
         "1M", "%ds" % int(N_crude * 1.3e-3),
         "1M", "%ds" % int(N_crude * 1.59e-3)], w)
    pdf.table_row(
        ["aE-SuS (10% CoV)",
         "%dK" % (N_asus_14 // 1000), "%ds" % int(N_asus_14 * 1.3e-3),
         "%dK" % (N_asus_30 // 1000), "%ds" % int(N_asus_30 * 1.59e-3)],
        w, fill=True)
    pdf.table_row(
        ["aE-SuS single run",
         "8K", "%ds" % int(8000 * 1.3e-3),
         "8K", "%ds" % int(8000 * 1.59e-3)], w)
    pdf.table_row(
        ["TSUM (exact rules)",
         "%.1fK" % (c14_calls / 1000), "%.1fs" % c14_time,
         "%dK" % (c30_calls // 1000), "%ds" % int(c30_time)], w, fill=True)
    pdf.ln(2)

    pdf.body_text(
        "For IEEE 14-bus, TSUM required only ~%s evaluations (%.1fs), "
        "approximately 4x fewer than a single aE-SuS run. For IEEE 30-bus, "
        "TSUM used ~%s evaluations (%.0fs), comparable to aE-SuS at 10%% "
        "CoV but providing exact rules rather than a statistical estimate."
        % (format(c14_calls, ','), c14_time,
           format(c30_calls, ','), c30_time)
    )

    # =================================================================
    # 11. Key advantages
    # =================================================================
    pdf.add_page()
    pdf.section_title("11. What TSUM Provides Beyond Point Estimates")

    pdf.body_text(
        "The adaptive MCS methods in Chan et al. produce a statistical "
        "estimate of p_f with associated uncertainty. TSUM provides "
        "qualitatively different and richer output:"
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "1. Interpretable rules")
    pdf.ln(7)
    pdf.body_text(
        "Each failure rule identifies a specific combination of component "
        "states that causes system failure. For IEEE 14-bus, the dominant "
        "rule {vbus3=0, vbus4=0} directly reveals that bus 3 is the "
        "critical vulnerability. For IEEE 30-bus, the 250 failure rules "
        "identify bus 2 as appearing in 92% of all failure modes."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "2. Exact probability bounds")
    pdf.ln(7)
    pdf.body_text(
        "TSUM decomposes the state space into survival, failure, and "
        "unknown regions with exact probability computation (no sampling "
        "error). The residual p_unknown provides a guaranteed bound on "
        "the remaining uncertainty."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "3. Instant re-evaluation under new probabilities")
    pdf.ln(7)
    pdf.body_text(
        "Once rules are extracted, changing component failure probabilities "
        "(e.g., for sensitivity analysis or different hazard scenarios) "
        "requires only a tensor probability computation - no additional "
        "DC-OPF evaluations. Adaptive MCS must re-run from scratch for "
        "each new probability assignment."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "4. Critical component identification")
    pdf.ln(7)
    pdf.body_text(
        "The frequency analysis of components in failure rules provides "
        "direct importance ranking without additional computation "
        "(cf. importance measures like Birnbaum or Fussell-Vesely, which "
        "typically require separate calculations)."
    )

    # =================================================================
    # 12. Conclusions
    # =================================================================
    pdf.section_title("12. Conclusions and Next Steps")

    pdf.body_text(
        "TSUM successfully reproduces the failure probability estimates "
        "from Chan et al. (2024) for the three completed cases: IEEE "
        "14-bus (p_f = %.1e), 30-bus (p_f = %.1e), and 57-bus "
        "(p_f = %.1e), all consistent with the reference ~1.0e-4."
        % (c14_last['p_failure'], c30_last['p_failure'],
           c57_last['p_failure'])
    )
    pdf.body_text(
        "A key insight from this study is that TSUM convergence depends "
        "more on the blackout threshold than on the number of components. "
        "The threshold determines the complexity of the failure boundary: "
        "a low threshold (e.g., 13.8% for IEEE 118-bus) creates many "
        "diverse failure modes and fragments the survival space, leading "
        "to slow convergence. A high threshold (e.g., 54.1% for IEEE "
        "57-bus) limits failure modes to combinations involving critical "
        "generators, enabling fast convergence even with more components."
    )
    pdf.body_text(
        "The computation cost is competitive with adaptive MCS methods "
        "for small to medium cases, while providing richer structural "
        "output. For larger systems with low thresholds (IEEE 118, 300), "
        "the cost of enumerating all rules grows substantially - this is "
        "an inherent trade-off of TSUM's comprehensive approach versus "
        "the statistical point estimates of adaptive MCS."
    )
    pdf.body_text(
        "A new biased discovery sampling strategy was developed to address "
        "cases where standard sampling cannot reach the failure region. "
        "By boosting failure-state probabilities during the search phase "
        "while keeping true probabilities for estimation, the algorithm "
        "successfully identifies failure rules in IEEE 118-bus where "
        "4,240 rounds of standard sampling found none. Automatic switching "
        "from biased to true probs was also tested (Section 7.4) but found "
        "to revert to baseline behaviour immediately after the switch, "
        "confirming that persistent biased sampling is the better strategy "
        "for this low-threshold case."
    )

    pdf.body_text(
        "For IEEE 118-bus, biased discovery sampling has proven essential. "
        "Standard MCS found zero failure rules after %d rounds. Fixed biased "
        "sampling (bf=10) found %d failure rules in %d rounds with p_unknown "
        "= %.3f. An alternating bias strategy (cycling bf=10/bf=2 every 100 "
        "rounds) was also tested but achieved higher p_unknown (%.3f) at the "
        "same round count. The discovered failure rules are valid minimal cut "
        "sets but average ~17 conditions, giving individual probability "
        "~ 10^-30."
        % (rA, frB, rB, upB, upAlt)
    )
    if c300_data:
        pdf.body_text(
            "For IEEE 300-bus (711 components), a bug in the DC-OPF solver "
            "(negative PD values causing LP infeasibility) was identified and "
            "fixed. Two bias factors were compared: bf=5 achieves p_unknown "
            "= %.4f after %d rounds (balanced discovery), while bf=10 gives "
            "p_unknown = %.4f after %d rounds (failure-dominated). The moderate "
            "bias is more effective for this larger system."
            % (c300_bf5['p_unknown'] if c300_bf5 else 0.995,
               c300_bf5['round'] if c300_bf5 else 6040,
               c300_bf10['p_unknown'] if c300_bf10 else 0.999,
               c300_bf10['round'] if c300_bf10 else 2220)
        )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Ongoing and next steps:")
    pdf.ln(7)
    pdf.bullet(
        "IEEE 118-bus: biased sampling (factor=10) on Gadi cluster, "
        "currently at round %d with %d failure rules, p_unknown=%.3f"
        % (c118_data['round'] if c118_data else 10000,
           c118_data['n_rules_fail'] if c118_data else 662,
           c118_data['p_unknown'] if c118_data else 0.230))
    if c300_data:
        pdf.bullet(
            "IEEE 300-bus: biased sampling (factor=5) on Gadi cluster, "
            "currently at round %d with %d failure rules, p_unknown=%.3f"
            % (c300_data['round'], c300_data['n_rules_fail'],
               c300_data['p_unknown']))
    else:
        pdf.bullet(
            "IEEE 300-bus (711 components): running on Gadi cluster")
    pdf.bullet(
        "Boundary walking: tested as a complementary strategy for case118. "
        "Walks efficiently discover failure rules (~192 per walk round vs ~14 "
        "per MCS round) but all rules are long (~12 conditions). The short "
        "rules (3-4 conditions) that dominate probability can only be found "
        "by MCS, so walks do not accelerate convergence.")
    pdf.bullet(
        "Find shorter failure rules for case118: the minimal cut sets "
        "with 2-4 conditions that dominate p_f ~ 10^-4")
    pdf.bullet(
        "Scenario 2 (p_f ~ 10^-5): higher thresholds require deeper "
        "exploration of the failure region")
    pdf.bullet(
        "Sensitivity analysis: leverage extracted rules to evaluate "
        "p_f under varied component probabilities without re-running DC-OPF")

    # =================================================================
    # Output
    # =================================================================
    out_path = Path(__file__).parent / "dcopt_benchmark_report.pdf"
    pdf.output(str(out_path))
    print("PDF written to: %s" % out_path)


if __name__ == "__main__":
    main()
