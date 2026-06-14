package players.neural;

import core.AbstractGameState;
import core.interfaces.IStateFeatureVector;
import core.interfaces.IStateHeuristic;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Uses the CRITIC head of a trained PPO policy as MCTS's state-evaluation
 * heuristic. Combined with rolloutLength=0, this makes MCTS evaluate leaves
 * directly via the learned value function with no rollouts at all -
 * sidestepping the rollout-bias / opponent-diversity problems that hurt the
 * policy-as-rollout approach.
 *
 * Loads the same weights JSON that NeuralRolloutPlayer uses, but ignores the
 * actor head and only runs trunk -> critic.
 *
 * The critic was trained from the deciding player's perspective (using the
 * same feature extractor) to predict expected outcome in [-1, 1]. The output
 * is linearly rescaled to [0, 1] and terminal states return WinOnlyHeuristic's
 * exact values, so this heuristic is scale-identical to the tuned baseline's
 * WinOnlyHeuristic and differs only on non-terminal (network-evaluated) states.
 */
public class CriticHeuristic implements IStateHeuristic {

    private final List<double[][]> hiddenW = new ArrayList<>();
    private final List<double[]> hiddenB = new ArrayList<>();
    private final double[][] criticW;
    private final double[] criticB;
    private final IStateFeatureVector features;

    public CriticHeuristic(String weightsPath, String featureClass) {
        try {
            this.features = (IStateFeatureVector)
                    Class.forName(featureClass).getConstructor().newInstance();
            JSONObject o = (JSONObject) new JSONParser().parse(new FileReader(weightsPath));
            for (Object ho : (JSONArray) o.get("hidden")) {
                JSONObject h = (JSONObject) ho;
                hiddenW.add(to2D((JSONArray) h.get("w")));
                hiddenB.add(to1D((JSONArray) h.get("b")));
            }
            JSONObject crit = (JSONObject) o.get("critic");
            this.criticW = to2D((JSONArray) crit.get("w"));
            this.criticB = to1D((JSONArray) crit.get("b"));
        } catch (Exception e) {
            throw new RuntimeException("Failed to init CriticHeuristic ("
                    + weightsPath + ", " + featureClass + ")", e);
        }
    }

    @Override
    public double evaluateState(AbstractGameState gs, int playerId) {
        // TERMINAL STATES: return the actual game result, never the network.
        // The critic only knows board patterns; it does not reliably recognise
        // that a game is over. Without this check, MCTS cannot distinguish an
        // actual win from a mid-game position (a won state might score 0.3
        // while a speculative branch scores 0.6), which is fatal in tactical
        // games like Connect4. This mirrors TAG's own heuristics (e.g.
        // WinOnlyHeuristic), which all dispatch on getPlayerResults() first.
        // Terminal values mirror WinOnlyHeuristic EXACTLY (same [0,1] scale), so
        // agents using this heuristic and agents using WinOnlyHeuristic back up
        // identical values for identical outcomes. The only remaining difference
        // between such agents is the non-terminal (network) evaluations.
        switch (gs.getPlayerResults()[playerId]) {
            case WIN_GAME:
                return 1.0;
            case DRAW_GAME:
            case TIMEOUT:
                return 0.5;
            case LOSE_GAME:
            case GAME_END:
            case DISQUALIFY:
                return 0.0;
            default:
                break; // GAME_ONGOING -> ask the critic
        }
        double[] obs = features.doubleVector(gs, playerId);
        double[] x = obs;
        for (int l = 0; l < hiddenW.size(); l++) {
            x = linear(x, hiddenW.get(l), hiddenB.get(l));
            for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0.0;
        }
        double[] v = linear(x, criticW, criticB);
        // The critic head was trained on returns in [-1, 1] (PyTAG rewards are
        // +/-1). Clamp to that range, then linearly rescale to [0, 1] so the
        // heuristic's scale matches WinOnlyHeuristic: 1 = certain win,
        // 0.5 = neutral/draw-ish, 0 = certain loss. A linear map preserves all
        // orderings; this is purely for scale consistency with the baseline.
        double raw = v[0];
        if (raw > 1.0) raw = 1.0;
        else if (raw < -1.0) raw = -1.0;
        return (raw + 1.0) / 2.0;
    }

    // Declared bounds must match the values actually returned. These mirror
    // WinOnlyHeuristic's bounds so MCTS normalisation treats both identically.
    @Override
    public double minValue() {
        return 0.0;
    }

    @Override
    public double maxValue() {
        return 1.0;
    }

    private static double[] linear(double[] x, double[][] w, double[] b) {
        double[] y = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            double s = b[i];
            double[] row = w[i];
            for (int j = 0; j < row.length; j++) s += row[j] * x[j];
            y[i] = s;
        }
        return y;
    }

    private static double[] to1D(JSONArray a) {
        double[] out = new double[a.size()];
        for (int i = 0; i < out.length; i++) out[i] = ((Number) a.get(i)).doubleValue();
        return out;
    }

    private static double[][] to2D(JSONArray a) {
        double[][] out = new double[a.size()][];
        for (int i = 0; i < out.length; i++) out[i] = to1D((JSONArray) a.get(i));
        return out;
    }
}
