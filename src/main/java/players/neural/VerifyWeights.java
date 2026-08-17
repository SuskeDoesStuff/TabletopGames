package players.neural;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Standalone weight-validation tool.
 *
 * Loads the JSON produced by export_weights.py and runs the same forward pass
 * (MLP -> actor logits) on the same fixed test inputs (zeros / ones / ramp)
 * that the Python exporter prints. If the Java output matches the Python
 * "reference outputs" block to ~4 decimal places, the Java forward pass and
 * weight layout are correct.
 *
 * Usage:
 *   java -cp target/TAG.jar players.neural.VerifyWeights /abs/path/weights.txt
 *
 * No game, no forward model, no feature extractor involved - this is pure math
 * on the loaded weights.
 */
public class VerifyWeights {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("usage: VerifyWeights <weights.txt>");
            return;
        }
        String path = args[0];

        JSONObject o = (JSONObject) new JSONParser().parse(new FileReader(path));

        List<double[][]> hiddenW = new ArrayList<>();
        List<double[]> hiddenB = new ArrayList<>();
        for (Object ho : (JSONArray) o.get("hidden")) {
            JSONObject h = (JSONObject) ho;
            hiddenW.add(to2D((JSONArray) h.get("w")));
            hiddenB.add(to1D((JSONArray) h.get("b")));
        }
        JSONObject act = (JSONObject) o.get("actor");
        double[][] actorW = to2D((JSONArray) act.get("w"));
        double[] actorB = to1D((JSONArray) act.get("b"));

        int obsDim = hiddenW.get(0)[0].length;
        System.out.println("meta: obs_dim=" + obsDim
                + " n_actions=" + actorW.length
                + " hidden_sizes=" + hiddenSizes(hiddenW));

        // Same three test inputs that export_weights.py prints reference outputs for.
        double[] zeros = new double[obsDim];
        double[] ones = new double[obsDim];
        for (int i = 0; i < obsDim; i++) ones[i] = 1.0;
        double[] ramp = new double[obsDim];                          // np.linspace(-1, 1, obs_dim)
        for (int i = 0; i < obsDim; i++) ramp[i] = -1.0 + 2.0 * i / (obsDim - 1);

        printRow("zeros", forward(zeros, hiddenW, hiddenB, actorW, actorB));
        printRow("ones",  forward(ones,  hiddenW, hiddenB, actorW, actorB));
        printRow("ramp",  forward(ramp,  hiddenW, hiddenB, actorW, actorB));
    }

    // obs -> [Linear, ReLU] x N -> actor Linear -> logits  (same as NeuralRolloutPlayer)
    private static double[] forward(double[] obs,
                                    List<double[][]> hW, List<double[]> hB,
                                    double[][] aW, double[] aB) {
        double[] x = obs;
        for (int l = 0; l < hW.size(); l++) {
            x = linear(x, hW.get(l), hB.get(l));
            for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0.0;
        }
        return linear(x, aW, aB);
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

    private static void printRow(String name, double[] logits) {
        int argmax = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[argmax]) argmax = i;
        StringBuilder sb = new StringBuilder();
        sb.append(name).append(": argmax=").append(argmax).append("  logits=[");
        for (int i = 0; i < logits.length; i++) {
            sb.append(String.format("%.4f", logits[i]));
            if (i < logits.length - 1) sb.append(", ");
        }
        sb.append("]");
        System.out.println(sb);
    }

    private static String hiddenSizes(List<double[][]> hW) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < hW.size(); i++) {
            sb.append(hW.get(i).length);
            if (i < hW.size() - 1) sb.append(", ");
        }
        return sb.append("]").toString();
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
