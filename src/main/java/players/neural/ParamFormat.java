package players.neural;

import core.AbstractPlayer;
import evaluation.optimisation.TunableParameters;
import players.PlayerParameters;
import players.mcts.MCTSParams;

// formatting for printing player settings during param logging
public final class ParamFormat {

    private ParamFormat() {}

    /** Multi-line block: name on its own line, then "indent param: value" lines. */
    public static String block(AbstractPlayer player, String indent) {
        StringBuilder sb = new StringBuilder();
        sb.append(player.toString());

        Object params = player.getParameters();
        if (params == null) {
            sb.append("\n").append(indent).append("(no parameters — reactive agent)");
            return sb.toString();
        }
        sb.append("  [").append(params.getClass().getSimpleName()).append("]");

        if (params instanceof TunableParameters<?> tunable) {
            for (String name : tunable.getParameterNames()) {
                Object value = tunable.getParameterValue(name);
                sb.append("\n").append(indent).append(name).append(": ").append(fmt(value));
            }
            // Append the live public-field values for the three params that
            // have previously been silently desynced from the registry.
            if (params instanceof PlayerParameters pp) {
                sb.append("\n").append(indent).append("(field) budget: ").append(pp.budget);
                sb.append("\n").append(indent).append("(field) budgetType: ").append(pp.budgetType);
            }
            if (params instanceof MCTSParams mp) {
                sb.append("\n").append(indent).append("(field) heuristic: ")
                  .append(mp.heuristic == null ? "null" : mp.heuristic.getClass().getSimpleName());
            }
        } else {
            sb.append("\n").append(indent).append("(non-tunable parameters)");
        }
        return sb.toString();
    }

    private static String fmt(Object value) {
        if (value == null) return "null";
        String s = value.toString();
        if (s.contains("$$Lambda") || s.contains("/0x")) {
            Class<?>[] ifaces = value.getClass().getInterfaces();
            return (ifaces.length > 0 ? ifaces[0].getSimpleName() : "lambda") + "(default)";
        }
        if (s.length() > 90) return s.substring(0, 87) + "...";
        return s;
    }
}
