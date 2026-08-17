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

// Heuristic that uses the critic head of the PPO policy for MCTS state evaluation for intermediate states and falls back to vanilla win/loss/draw settings for terminal states.
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
 
        double raw = v[0];
        if (raw > 1.0) raw = 1.0;
        else if (raw < -1.0) raw = -1.0;
        return (raw + 1.0) / 2.0;
    }
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
