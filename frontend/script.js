/* ============================================================
   script.js — Frontend logic for Chess Analysis Tools
   ============================================================ */

const API_BASE = "http://127.0.0.1:5000/api";

// ── Utility ──────────────────────────────────────────────────

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function show(el) { if (typeof el === "string") el = $(el); if (el) el.style.display = "block"; }
function hide(el) { if (typeof el === "string") el = $(el); if (el) el.style.display = "none"; }

function clearErrors(prefix) {
    $$(`[id^="err-"]`).forEach(e => e.textContent = "");
    $$(`input.error, select.error, textarea.error`).forEach(e => e.classList.remove("error"));
}

function setError(field, msg) {
    const input = $(`#${field}`);
    const errEl = $(`#err-${field}`);
    if (input) input.classList.add("error");
    if (errEl) errEl.textContent = msg;
}

function showAlert(id, msg) {
    const el = $(id);
    if (el) { el.textContent = msg; show(el); }
}

function hideAlert(id) { hide(id); }

async function apiCall(endpoint, body) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw { status: res.status, data };
    return data;
}

// ── Tabs ─────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    $$(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab;
            // Deactivate all in same group
            const parent = btn.closest(".tabs");
            parent.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            // Show matching content
            const container = parent.parentElement;
            container.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
            const target = container.querySelector(`#tab-${tabId}`);
            if (target) target.classList.add("active");
        });
    });
});

// ── Validation helpers ───────────────────────────────────────

function validateInt(field, label, lo, hi) {
    const val = $(`#${field}`)?.value?.trim();
    if (!val) { setError(field, `${label} is required.`); return null; }
    const n = parseInt(val, 10);
    if (isNaN(n)) { setError(field, `${label} must be a number.`); return null; }
    if (lo !== undefined && n < lo) { setError(field, `Must be ≥ ${lo}.`); return null; }
    if (hi !== undefined && n > hi) { setError(field, `Must be ≤ ${hi}.`); return null; }
    return n;
}

function validateRequired(field, label) {
    const val = $(`#${field}`)?.value?.trim();
    if (!val) { setError(field, `${label} is required.`); return null; }
    return val;
}

// =============================================================
//  OPENING RECOMMENDER
// =============================================================

const openingForm = $("#opening-form");
if (openingForm) {
    openingForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearErrors();
        hideAlert("#opening-error");
        hide("#opening-result");

        let valid = true;
        const your_elo     = validateInt("your_elo",       "Your ELO",        100, 3500);
        const opponent_elo = validateInt("opponent_elo",    "Opponent ELO",    100, 3500);
        const color        = $("#play_color")?.value;
        const base_time    = validateInt("open_base_time",  "Base time",       0,   10800);

        let increment = parseInt($("#open_increment")?.value || "0", 10);
        if (isNaN(increment)) increment = 0;

        const topN = parseInt($("#top_n")?.value || "10", 10);

        if (your_elo === null || opponent_elo === null || base_time === null) return;

        const btn = $("#btn-opening");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Analyzing…';

        try {
            const result = await apiCall("/opening", {
                your_elo, opponent_elo, color, base_time, increment,
            });

            const openings = topN > 0 ? result.openings.slice(0, topN) : result.openings;

            // Hero
            const best = openings[0];
            const bestPct = (best.win_probability * 100).toFixed(1) + "%";
            const hero = $("#opening-hero");
            hero.className = "result-hero " + (best.win_probability > 0.55 ? "safe" : best.win_probability > 0.45 ? "neutral" : "danger");
            $("#opening-best-prob").textContent = bestPct;
            $("#opening-verdict").textContent = `Best: ${best.opening_name} (${best.eco_code})`;

            // Summary detail
            $("#opening-detail").innerHTML = `
                <div class="detail-item"><div class="dl">Your ELO</div><div class="dv">${result.your_elo}</div></div>
                <div class="detail-item"><div class="dl">Opponent ELO</div><div class="dv">${result.opponent_elo}</div></div>
                <div class="detail-item"><div class="dl">Color</div><div class="dv" style="text-transform:capitalize;">${result.color}</div></div>
                <div class="detail-item"><div class="dl">Time Control</div><div class="dv">${result.base_time}s + ${result.increment}s</div></div>
                <div class="detail-item"><div class="dl">Showing</div><div class="dv">${openings.length} of ${result.total_openings} openings</div></div>
            `;

            // Opening table
            const tbody = $("#opening-body");
            tbody.innerHTML = "";
            openings.forEach((o, i) => {
                const wp = (o.win_probability * 100).toFixed(1);
                let barColor;
                if (o.win_probability > 0.55)      barColor = "var(--green)";
                else if (o.win_probability > 0.45)  barColor = "var(--orange)";
                else                                barColor = "var(--red)";

                tbody.innerHTML += `
                    <tr>
                        <td>${i + 1}</td>
                        <td><strong>${o.eco_code}</strong></td>
                        <td>${o.opening_name}</td>
                        <td>
                            <div style="display:flex;align-items:center;gap:6px;">
                                <div style="width:60px;height:8px;border-radius:4px;background:var(--surface-2);overflow:hidden;">
                                    <div style="width:${wp}%;height:100%;background:${barColor};border-radius:4px;"></div>
                                </div>
                                <span>${wp}%</span>
                            </div>
                        </td>
                        <td>
                            ${o.win_probability > 0.55 ? '<span class="blunder-badge low">Strong</span>' :
                              o.win_probability > 0.45 ? '<span class="blunder-badge medium">Neutral</span>' :
                              '<span class="blunder-badge high">Weak</span>'}
                        </td>
                    </tr>`;
            });

            show("#opening-result");
            $("#opening-result").classList.add("visible");

        } catch (err) {
            const msg = err.data?.errors?.join(", ") || err.data?.error || "Something went wrong.";
            showAlert("#opening-error", msg);
        } finally {
            btn.disabled = false;
            btn.innerHTML = "Recommend Openings";
        }
    });

    // Reset
    $("#btn-opening-reset")?.addEventListener("click", () => {
        openingForm.reset();
        clearErrors();
        hide("#opening-result");
        hideAlert("#opening-error");
    });
}

// =============================================================
//  BLUNDER PREDICTOR — PGN Game
// =============================================================

const pgnForm = $("#pgn-form");
if (pgnForm) {
    pgnForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearErrors();
        hideAlert("#pgn-error");
        hide("#pgn-result");

        const pgn_text = validateRequired("pgn_text", "PGN text");
        if (!pgn_text) return;

        // Sanity check
        if (pgn_text.length < 10) {
            setError("pgn_text", "PGN seems too short. Paste a full game.");
            return;
        }

        const body = { pgn: pgn_text };
        const we = $("#pgn_white_elo")?.value?.trim();
        const be = $("#pgn_black_elo")?.value?.trim();
        if (we) body.white_elo = parseInt(we, 10);
        if (be) body.black_elo = parseInt(be, 10);

        const btn = $("#btn-pgn");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Analyzing…';

        try {
            const result = await apiCall("/predict-pgn", body);

            // Hero
            const hero = $("#pgn-hero");
            const bc = result.blunders_found;
            hero.className = "result-hero " + (bc > 5 ? "danger" : bc > 0 ? "neutral" : "safe");
            $("#pgn-blunder-count").textContent = bc;
            $("#pgn-verdict").textContent = bc === 0
                ? "Clean game — no blunders detected"
                : `${bc} blunder${bc > 1 ? "s" : ""} found (cp loss ≥ ${result.blunder_threshold}) out of ${result.total_moves} moves`;

            // Summary
            $("#pgn-detail").innerHTML = `
                <div class="detail-item"><div class="dl">Total Moves</div><div class="dv">${result.total_moves}</div></div>
                <div class="detail-item"><div class="dl">Blunders Found</div><div class="dv">${bc}</div></div>
                <div class="detail-item"><div class="dl">White ELO</div><div class="dv">${result.white_elo}</div></div>
                <div class="detail-item"><div class="dl">Black ELO</div><div class="dv">${result.black_elo}</div></div>
                <div class="detail-item"><div class="dl">Engine Depth</div><div class="dv">${result.engine_depth}</div></div>
                <div class="detail-item"><div class="dl">Threshold</div><div class="dv">${result.blunder_threshold} cp</div></div>
            `;

            // Move table
            const tbody = $("#pgn-moves-body");
            tbody.innerHTML = "";
            result.moves.forEach(m => {
                const cpLoss = m.cp_loss;
                let badge = "";
                if (m.is_blunder)       badge = `<span class="blunder-badge high">BLUNDER (${cpLoss} cp)</span>`;
                else if (cpLoss >= 100)  badge = `<span class="blunder-badge medium">Inaccuracy</span>`;
                else                     badge = `<span class="blunder-badge low">OK</span>`;

                // Format eval as +/- with fixed decimals
                const fmtEval = (v) => {
                    if (v >= 10000) return "#";
                    if (v <= -10000) return "#";
                    const sign = v >= 0 ? "+" : "";
                    return sign + (v / 100).toFixed(2);
                };

                tbody.innerHTML += `
                    <tr class="${m.is_blunder ? 'blunder-row' : ''}">
                        <td>${m.move_number}</td>
                        <td><strong>${m.move_san}</strong></td>
                        <td>${m.side}</td>
                        <td>${fmtEval(m.eval_before)}</td>
                        <td>${fmtEval(m.eval_after)}</td>
                        <td>${cpLoss}</td>
                        <td>${badge}</td>
                    </tr>`;
            });

            show("#pgn-result");
            $("#pgn-result").classList.add("visible");

        } catch (err) {
            const msg = err.data?.errors?.join(", ") || err.data?.error || "Failed to analyze PGN.";
            showAlert("#pgn-error", msg);
        } finally {
            btn.disabled = false;
            btn.innerHTML = "Analyze Game";
        }
    });

    // Reset
    $("#btn-pgn-reset")?.addEventListener("click", () => {
        pgnForm.reset();
        clearErrors();
        hide("#pgn-result");
        hideAlert("#pgn-error");
    });
}

// =============================================================
//  ELO CALCULATOR — Single Game
// =============================================================

const eloForm = $("#elo-form");
if (eloForm) {
    eloForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearErrors();
        hideAlert("#elo-error");
        hide("#elo-result");

        const your_rating    = validateInt("your_rating",    "Your rating",    100, 3500);
        const opponent_rating = validateInt("opponent_rating","Opponent rating", 100, 3500);
        const outcome         = validateRequired("outcome",   "Outcome");
        let   games_played    = parseInt($("#games_played")?.value || "30", 10);
        if (isNaN(games_played) || games_played < 0) games_played = 30;

        const ageRaw = $("#player_age")?.value?.trim();
        let age = null;
        if (ageRaw) {
            age = parseInt(ageRaw, 10);
            if (isNaN(age) || age < 5 || age > 120) {
                setError("player_age", "Age must be between 5 and 120.");
                return;
            }
        }

        if (your_rating === null || opponent_rating === null || !outcome) return;

        const btn = $("#btn-elo");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Calculating…';

        try {
            const r = await apiCall("/elo", {
                your_rating, opponent_rating, outcome, games_played,
                ...(age !== null ? { age } : {}),
            });

            // Arrow
            const arrowEl = $("#elo-arrow");
            if (r.change > 0)      { arrowEl.textContent = "▲"; arrowEl.className = "arrow up"; }
            else if (r.change < 0) { arrowEl.textContent = "▼"; arrowEl.className = "arrow down"; }
            else                   { arrowEl.textContent = "●"; arrowEl.className = "arrow same"; }

            $("#elo-old").textContent = `Was: ${r.old_rating}`;
            $("#elo-new").textContent = r.new_rating;

            const deltaEl = $("#elo-delta");
            deltaEl.textContent = (r.change > 0 ? "+" : "") + r.change;
            deltaEl.className = "rc-delta " + (r.change > 0 ? "pos" : r.change < 0 ? "neg" : "");

            // Detail
            $("#elo-detail").innerHTML = `
                <div class="detail-item"><div class="dl">Outcome</div><div class="dv" style="text-transform:capitalize;">${r.outcome}</div></div>
                <div class="detail-item"><div class="dl">Opponent</div><div class="dv">${r.opponent_rating}</div></div>
                <div class="detail-item"><div class="dl">Win Probability</div><div class="dv">${r.win_probability}%</div></div>
                <div class="detail-item"><div class="dl">K-Factor</div><div class="dv">${r.k_factor}</div></div>
                <div class="detail-item"><div class="dl">K Reason</div><div class="dv" style="text-transform:capitalize;">${r.k_reason}</div></div>
                <div class="detail-item"><div class="dl">Expected Score</div><div class="dv">${r.expected_score}</div></div>
                <div class="detail-item"><div class="dl">Actual Score</div><div class="dv">${r.actual_score}</div></div>
                ${r.age !== null && r.age !== undefined ? `<div class="detail-item"><div class="dl">Age</div><div class="dv">${r.age}</div></div>` : ""}
            `;

            show("#elo-result");
            $("#elo-result").classList.add("visible");

        } catch (err) {
            const msg = err.data?.errors?.join(", ") || err.data?.error || "Calculation failed.";
            showAlert("#elo-error", msg);
        } finally {
            btn.disabled = false;
            btn.innerHTML = "Calculate";
        }
    });

    // Reset
    $("#btn-elo-reset")?.addEventListener("click", () => {
        eloForm.reset();
        clearErrors();
        hide("#elo-result");
        hideAlert("#elo-error");
    });
}

// =============================================================
//  ELO CALCULATOR — Multi-Game Session
// =============================================================

let sessionGames = [];

function renderSessionChips() {
    const list = $("#session-games-list");
    if (!list) return;
    if (sessionGames.length === 0) {
        list.innerHTML = '<span style="color:var(--muted);font-size:.85rem;">No games added yet.</span>';
        return;
    }
    list.innerHTML = sessionGames.map((g, i) => {
        const oc = g.outcome;
        const color = oc === "win" ? "var(--green)" : oc === "loss" ? "var(--red)" : "var(--orange)";
        const name = g.opponent_name || `Opp ${i + 1}`;
        return `<span class="game-chip">
            <strong style="color:${color}">${oc.toUpperCase()}</strong>
            vs ${name} (${g.opponent_rating})
            <span class="remove-game" data-idx="${i}">&times;</span>
        </span>`;
    }).join("");

    // Remove buttons
    list.querySelectorAll(".remove-game").forEach(btn => {
        btn.addEventListener("click", () => {
            sessionGames.splice(parseInt(btn.dataset.idx), 1);
            renderSessionChips();
        });
    });
}

// Add game button
document.addEventListener("DOMContentLoaded", () => {
    renderSessionChips();

    $("#btn-add-game")?.addEventListener("click", () => {
        const rating = parseInt($("#sg_opp_rating")?.value, 10);
        const outcome = $("#sg_outcome")?.value;
        const name = $("#sg_opp_name")?.value?.trim();

        if (!rating || rating < 100 || rating > 3500) {
            alert("Enter a valid opponent rating (100–3500).");
            return;
        }

        sessionGames.push({
            opponent_rating: rating,
            outcome: outcome,
            opponent_name: name || "",
        });

        // Clear inputs
        $("#sg_opp_rating").value = "";
        $("#sg_opp_name").value = "";
        renderSessionChips();
    });

    // Calculate session
    $("#btn-session")?.addEventListener("click", async () => {
        clearErrors();
        hideAlert("#session-error");
        hide("#session-result");

        const rating = validateInt("session_rating", "Starting rating", 100, 3500);
        if (rating === null) return;

        let gp = parseInt($("#session_games_played")?.value || "30", 10);
        if (isNaN(gp) || gp < 0) gp = 30;

        const sessionAgeRaw = $("#session_age")?.value?.trim();
        let sessionAge = null;
        if (sessionAgeRaw) {
            sessionAge = parseInt(sessionAgeRaw, 10);
            if (isNaN(sessionAge) || sessionAge < 5 || sessionAge > 120) {
                showAlert("#session-error", "Age must be between 5 and 120.");
                return;
            }
        }

        if (sessionGames.length === 0) {
            showAlert("#session-error", "Add at least one game first.");
            return;
        }

        const btn = $("#btn-session");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span> Calculating…';

        try {
            const r = await apiCall("/elo/session", {
                your_rating: rating,
                games_played: gp,
                ...(sessionAge !== null ? { age: sessionAge } : {}),
                games: sessionGames,
            });

            // Hero
            const hero = $("#session-hero");
            hero.className = "result-hero " + (r.total_change > 0 ? "safe" : r.total_change < 0 ? "danger" : "neutral");
            $("#session-final").textContent = r.final_rating;
            const sign = r.total_change > 0 ? "+" : "";
            $("#session-verdict").textContent = `${sign}${r.total_change} from ${r.starting_rating}  —  ${r.summary.wins}W / ${r.summary.draws}D / ${r.summary.losses}L`;

            // Summary
            $("#session-summary").innerHTML = `
                <div class="detail-item"><div class="dl">Start</div><div class="dv">${r.starting_rating}</div></div>
                <div class="detail-item"><div class="dl">Final</div><div class="dv">${r.final_rating}</div></div>
                <div class="detail-item"><div class="dl">Net Change</div><div class="dv" style="color:${r.total_change >= 0 ? 'var(--green)' : 'var(--red)'}">${sign}${r.total_change}</div></div>
                <div class="detail-item"><div class="dl">Games</div><div class="dv">${r.games.length}</div></div>
            `;

            // Per-game breakdown
            const detail = $("#session-games-detail");
            detail.innerHTML = r.games.map(g => {
                const oc = g.outcome;
                const d = g.change > 0 ? "+" + g.change : g.change;
                return `<div class="session-game">
                    <div class="sg-num">#${g.game_number}</div>
                    <div>vs ${g.opponent_name} (${g.opponent_rating})</div>
                    <div class="sg-outcome ${oc}">${oc}</div>
                    <div style="font-weight:700;color:${g.change >= 0 ? 'var(--green)' : 'var(--red)'}">${d}</div>
                </div>`;
            }).join("");

            show("#session-result");
            $("#session-result").classList.add("visible");

        } catch (err) {
            const msg = err.data?.errors?.join(", ") || err.data?.error || "Session calculation failed.";
            showAlert("#session-error", msg);
        } finally {
            btn.disabled = false;
            btn.innerHTML = "Calculate Session";
        }
    });

    // Clear session
    $("#btn-session-reset")?.addEventListener("click", () => {
        sessionGames = [];
        renderSessionChips();
        clearErrors();
        hide("#session-result");
        hideAlert("#session-error");
        if ($("#session_rating")) $("#session_rating").value = "";
    });
});
