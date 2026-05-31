package players.neural;

import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * MCTS variant that uses the CRITIC head as the leaf-evaluation heuristic
 * instead of running rollouts. Sets rolloutLength=0 so MCTS skips rollouts
 * entirely and reads the value directly from the learned critic.
 *
 * Sister class to NeuralMCTSPlayer (which uses the actor as a rollout policy).
 * These represent the two principled ways to inject a trained PPO net into
 * MCTS - "policy steers playouts" vs "value evaluates leaves" - and can be
 * run side by side in a tournament.
 *
 *   new NeuralCriticMCTSPlayer("/abs/diamant_weights.txt", "games.diamant.DiamantFeatures")
 */
public class NeuralCriticMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralCriticMCTSPlayer(String weightsPath, String featureClass) {
        super(buildParams(weightsPath, featureClass), "MCTS-CriticHeuristic");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams buildParams(String weightsPath, String featureClass) {
        MCTSParams params = new MCTSParams();
        // The critic replaces rollouts entirely.
        params.heuristic = new CriticHeuristic(weightsPath, featureClass);
        // rolloutLength=0 skips the rollout loop; MCTS evaluates the leaf state
        // directly via params.heuristic (line 908 of SingleTreeNode.rollout).
        params.setParameterValue("rolloutLength", 0);
        return params;
    }

    @Override
    public NeuralCriticMCTSPlayer copy() {
        NeuralCriticMCTSPlayer c = new NeuralCriticMCTSPlayer(weightsPath, featureClass);
        c.setForwardModel(getForwardModel());
        if (getParameters() != null) c.setBudget(getParameters().budget);
        c.setName(toString());
        return c;
    }
}
