package players.neural;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.Game;
import games.GameType;
import games.sushigo.SGFeatures;
import players.simple.RandomPlayer;

import java.util.List;
import java.util.Random;

/**
 * Parity harness for SushiGo features.
 *
 * Plays a 2-player SushiGo game forward a handful of random moves to reach a
 * non-trivial mid-round state (some cards played, some in hand), then for the
 * current player prints BOTH:
 *   1. the JSON observation (what getObservationJson emits, what the Python
 *      wrapper consumes), and
 *   2. the doubleVector output (the Java vector path used at inference),
 *      with its length.
 *
 * To confirm parity: paste the printed JSON into the Python snippet (see
 * SUSHIGO_QUICKSTART.md), run SushiGoWrapper.process_json_obs on it, and check
 * the resulting numpy array equals the doubleVector printed here, element by
 * element. They should match to floating-point precision.
 *
 * Run:
 *   java -cp target/TAG.jar players.neural.SGParityCheck [seed] [nMoves]
 */
public class SGParityCheck {

    public static void main(String[] args) {
        long seed = args.length > 0 ? Long.parseLong(args[0]) : 42L;
        int nMoves = args.length > 1 ? Integer.parseInt(args[1]) : 6;

        // 2-player SushiGo
        Game game = GameType.SushiGo.createGameInstance(2, seed);
        AbstractGameState gs = game.getGameState();
        Random rng = new Random(seed);

        // advance a few moves with random legal actions to reach a mid-round state
        for (int m = 0; m < nMoves && gs.isNotTerminal(); m++) {
            List<AbstractAction> actions =
                    game.getForwardModel().computeAvailableActions(gs);
            if (actions.isEmpty()) break;
            AbstractAction a = actions.get(rng.nextInt(actions.size()));
            game.getForwardModel().next(gs, a.copy());
        }

        int pid = gs.getCurrentPlayer();
        SGFeatures feat = new SGFeatures();

        String json = feat.getObservationJson(gs, pid);
        double[] vec = feat.doubleVector(gs, pid);

        System.out.println("=== SushiGo parity check ===");
        System.out.println("seed=" + seed + "  movesPlayed=" + nMoves
                + "  round=" + gs.getRoundCounter()
                + "  currentPlayer=" + pid
                + "  nPlayers=" + gs.getNPlayers());
        System.out.println();
        System.out.println("--- JSON observation (feed this to Python process_json_obs) ---");
        System.out.println(json);
        System.out.println();
        System.out.println("--- doubleVector (length " + vec.length + ") ---");
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vec.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(trim(vec[i]));
        }
        sb.append("]");
        System.out.println(sb);
        System.out.println();
        System.out.println("Expected length at 2 players: 147");
        if (vec.length != 147)
            System.out.println("WARNING: length is " + vec.length + ", not 147 — check N_PLAYERS / layout.");
    }

    private static String trim(double d) {
        if (d == Math.rint(d)) return String.valueOf((long) d);  // whole numbers clean
        return String.format("%.4f", d);
    }
}
