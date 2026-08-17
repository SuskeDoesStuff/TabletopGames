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
import java.util.Random;

// Standalone Neural Player
public class NeuralRolloutPlayer extends AbstractPlayer {

    private final List<double[][]> hiddenW = new ArrayList<>();
    private final List<double[]> hiddenB = new ArrayList<>();
    private final double[][] actorW;
    private final double[] actorB;
    private final IStateFeatureVector features;
    private final boolean greedy;
    private final Random rng;

    private final String weightsPath;
    private final String featureClass;
    private transient ActionTreeNode treeRoot = null;

    // Two-arg ctor: defaults to sampling (the recommended setting).
    public NeuralRolloutPlayer(String weightsPath, String featureClass) {
        this(weightsPath, featureClass, false);
    }

    // Three-arg ctor: explicit greedy flag for argmax A/B testing.
    public NeuralRolloutPlayer(String weightsPath, String featureClass, boolean greedy) {
        super(new PlayerParameters(), "NeuralRollout" + (greedy ? "-Greedy" : "-Sample"));
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
        this.greedy = greedy;
        this.rng = new Random(getParameters().getRandomSeed());
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

        int n = Math.min(leaves.size(), logits.length);
        int chosen = greedy ? argmaxLegal(leaves, logits, n)
                            : sampleSoftmaxLegal(leaves, logits, n);
        if (chosen >= 0 && leaves.get(chosen).getAction() != null) {
            return leaves.get(chosen).getAction();
        }
        return actions.get(0); // safety fallback (should not trigger)
    }

    // Greedy: highest-logit legal leaf.
    private static int argmaxLegal(List<ActionTreeNode> leaves, double[] logits, int n) {
        int best = -1;
        double bestLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (leaves.get(i).getValue() == 1 && logits[i] > bestLogit) {
                bestLogit = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Softmax sample over LEGAL leaves only. Numerically stable: subtract max
    // logit before exp(), then walk a uniform * sum threshold.
    private int sampleSoftmaxLegal(List<ActionTreeNode> leaves, double[] logits, int n) {
        double maxL = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            if (leaves.get(i).getValue() == 1 && logits[i] > maxL) maxL = logits[i];
        }
        if (maxL == Double.NEGATIVE_INFINITY) return -1; // no legal leaf

        double[] probs = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            if (leaves.get(i).getValue() == 1) {
                probs[i] = Math.exp(logits[i] - maxL);
                sum += probs[i];
            }
        }
        double threshold = rng.nextDouble() * sum;
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            if (leaves.get(i).getValue() != 1) continue;
            acc += probs[i];
            if (acc >= threshold) return i;
        }
        return -1; // unreachable except for FP edge cases
    }

    // obs -> [Linear, ReLU] x N -> actor Linear -> logits
    private double[] forwardActor(double[] obs) {
        double[] x = obs;
        for (int l = 0; l < hiddenW.size(); l++) {
            x = linear(x, hiddenW.get(l), hiddenB.get(l));
            for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0.0;
        }
        return linear(x, actorW, actorB);
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

    @Override
    public NeuralRolloutPlayer copy() {
        return new NeuralRolloutPlayer(weightsPath, featureClass, greedy);
    }
}
