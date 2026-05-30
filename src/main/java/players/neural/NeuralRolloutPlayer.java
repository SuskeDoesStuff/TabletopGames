package players.neural;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateFeatureVector;
import core.interfaces.ITreeActionSpace;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import players.PlayerParameters;
import utilities.ActionTreeNode;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * A GAME-AGNOSTIC rollout policy backed by a PyTAG PPO (PPONet) MLP.
 *
 * Nothing here is game-specific. Two things are injected at construction:
 *   - weightsPath:        the JSON produced by export_weights.py
 *   - featureExtractor:   the fully-qualified IStateFeatureVector class name
 *                         (MUST match the one PyTAG used to train this game,
 *                          e.g. games.tictactoe.TTTFeatures, games.diamant.DiamantFeatures,
 *                          games.sushigo.SGFeatures)
 *
 * The action mapping uses the SAME ITreeActionSpace machinery PyTAG used during
 * training (initActionTree -> updateActionTree -> getLeafNodes), so the network's
 * output index i lines up with leaf i exactly. No per-game action code, no casts.
 *
 * Only handles MLP policies (any number of hidden layers). For CNN/LSTM policies,
 * use ONNX instead.
 */
public class NeuralRolloutPlayer extends AbstractPlayer {

    private final List<double[][]> hiddenW = new ArrayList<>();
    private final List<double[]> hiddenB = new ArrayList<>();
    private final double[][] actorW;
    private final double[] actorB;
    private final IStateFeatureVector features;

    // kept so copy() can re-create an identical instance
    private final String weightsPath;
    private final String featureClass;

    // cached action tree (structure is fixed for the whole game)
    private transient ActionTreeNode treeRoot = null;

    public NeuralRolloutPlayer(String weightsPath, String featureClass) {
        super(new PlayerParameters(), "NeuralRollout");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
        try {
            this.features = (IStateFeatureVector)
                    Class.forName(featureClass).getConstructor().newInstance();
            JSONObject o = (JSONObject) new JSONParser().parse(new FileReader(weightsPath));
            for (Object ho : (JSONArray) o.get("hidden")) {
                JSONObject h = (JSONObject) ho;
                hiddenW.add(to2D((JSONArray) h.get("w")));
                hiddenB.add(to1D((JSONArray) h.get("b")));
            }
            JSONObject act = (JSONObject) o.get("actor");
            this.actorW = to2D((JSONArray) act.get("w"));
            this.actorB = to1D((JSONArray) act.get("b"));
        } catch (Exception e) {
            throw new RuntimeException("Failed to init NeuralRolloutPlayer (" +
                    weightsPath + ", " + featureClass + ")", e);
        }
    }

    @Override
    public AbstractAction _getAction(AbstractGameState state, List<AbstractAction> actions) {
        ITreeActionSpace fm = (ITreeActionSpace) getForwardModel();
        if (treeRoot == null) treeRoot = fm.initActionTree(state);
        fm.updateActionTree(treeRoot, state);
        List<ActionTreeNode> leaves = treeRoot.getLeafNodes();

        double[] obs = features.doubleVector(state, state.getCurrentPlayer());
        double[] logits = forwardActor(obs);

        int best = -1;
        double bestLogit = Double.NEGATIVE_INFINITY;
        int n = Math.min(leaves.size(), logits.length);
        for (int i = 0; i < n; i++) {
            // getValue()==1 marks a legal leaf -> implicit action masking
            if (leaves.get(i).getValue() == 1 && logits[i] > bestLogit) {
                bestLogit = logits[i];
                best = i;
            }
        }
        if (best >= 0 && leaves.get(best).getAction() != null) {
            return leaves.get(best).getAction();
        }
        return actions.get(0); // safety fallback (should not trigger)
    }

    // obs -> [Linear, ReLU] x N -> actor Linear (raw logits)
    private double[] forwardActor(double[] obs) {
        double[] x = obs;
        for (int l = 0; l < hiddenW.size(); l++) {
            x = linear(x, hiddenW.get(l), hiddenB.get(l));
            for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0.0; // ReLU
        }
        return linear(x, actorW, actorB);
    }

    // y = W x + b  (W stored [out][in], exactly as PyTorch saves it -> no transpose)
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

    @Override
    public NeuralRolloutPlayer copy() {
        return new NeuralRolloutPlayer(weightsPath, featureClass);
    }
}
