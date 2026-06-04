package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * MCTS variant that uses the trained policy as the rollout strategy AND the
 * trained critic as the leaf-value heuristic, with a short rollout length so
 * neither dominates.
 *
 * Architecturally: classical UCT-MCTS with both neural injections layered.
 * Selection still uses pure UCT (no policy prior) — this is NOT AlphaZero.
 *
 * Each MCTS iteration:
 *   1. Selection walks the tree via UCT.
 *   2. Expansion adds a new leaf.
 *   3. Rollout plays {@code rolloutLength} plies using the trained policy.
 *   4. The critic evaluates the resulting state and that value is backed up.
 *
 * The motivation: each pure injection has a distinct failure mode on Diamant.
 *   - Pure policy rollout to terminal: bias accumulates over many policy plies.
 *   - Pure critic with rolloutLength=0: critic queried on the leaf state
 *     directly, often OOD for states MCTS reaches by exploring beyond the
 *     policy's natural distribution.
 *
 * Short policy rollout + critic compromise: a few policy plies move the state
 * into the trained distribution (where the critic was trained), then the
 * critic gives the value. Both biases still present, but neither extreme.
 *
 *   new NeuralBothMCTSPlayer("/abs/weights.txt", "games.diamant.DiamantFeatures")
 */
public class NeuralBothMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralBothMCTSPlayer(String weightsPath, String featureClass) {
        super(buildParams(weightsPath, featureClass), "MCTS-NeuralBoth");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams buildParams(String weightsPath, String featureClass) {
        MCTSParams params = new MCTSParams();

        // Policy as rollout strategy (same wiring as NeuralMCTSPlayer)
        params.setParameterValue("rolloutType", MCTSEnums.Strategies.CLASS);
        params.setParameterValue("rolloutClass",
                "{\"class\":\"players.neural.NeuralRolloutPlayer\",\"args\":[\""
                        + weightsPath + "\",\"" + featureClass + "\"]}");

        // Critic as leaf-value heuristic (same wiring as NeuralCriticMCTSPlayer)
        params.heuristic = new CriticHeuristic(weightsPath, featureClass);

        // Short rollout: enough plies for the state to leave the immediate
        // leaf neighbourhood without accumulating much policy bias. Tune as
        // desired; 3 - 10 are reasonable starting values.
        params.setParameterValue("rolloutLength", 5);

        return params;
    }

    @Override
    public NeuralBothMCTSPlayer copy() {
        NeuralBothMCTSPlayer c = new NeuralBothMCTSPlayer(weightsPath, featureClass);
        c.setForwardModel(getForwardModel());
        if (getParameters() != null) c.setBudget(getParameters().budget);
        c.setName(toString());
        return c;
    }
}
