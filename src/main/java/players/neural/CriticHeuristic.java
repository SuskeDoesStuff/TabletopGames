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
 * same feature extractor) to predict expected outcome in [-1, 1]. We feed the
 * obs from the evaluated player's perspective and return that prediction.
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
        double[] obs = features.doubleVector(gs, playerId);
        double[] x = obs;
        for (int l = 0; l < hiddenW.size(); l++) {
            x = linear(x, hiddenW.get(l), hiddenB.get(l));
            for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0.0;
        }
        double[] v = linear(x, criticW, criticB);
        // critic head outputs a single scalar; clamp lightly to stay in [-1, 1]
        // since MCTS uses minValue/maxValue for normalisation.
        double raw = v[0];
        if (raw > 1.0) raw = 1.0;
        else if (raw < -1.0) raw = -1.0;
        return raw;
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
