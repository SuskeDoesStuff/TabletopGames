package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

// Combined neural injection player with policy guiding the rollouts and critic heuristic for state evaluation.
public class NeuralBothMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralBothMCTSPlayer(String weightsPath, String featureClass) {
        this(new MCTSParams(), weightsPath, featureClass);
    }

    public NeuralBothMCTSPlayer(MCTSParams baseParams, String weightsPath, String featureClass) {
        super(applyBoth(baseParams, weightsPath, featureClass), "MCTS-NeuralBoth");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams applyBoth(MCTSParams base, String weights, String features) {
        MCTSParams modified = (MCTSParams) base.copy();
        // Policy as rollout strategy
        modified.setParameterValue("rolloutType", MCTSEnums.Strategies.CLASS);
        modified.setParameterValue("rolloutClass",
                "{\"class\":\"players.neural.NeuralRolloutPlayer\",\"args\":[\""
                        + weights + "\",\"" + features + "\"]}");
        // Critic as a heuristic
        modified.setParameterValue("heuristic", new CriticHeuristic(weights, features));

        // Sanity check: fail loudly if the critic did not survive.
        if (!(modified.heuristic instanceof CriticHeuristic))
            throw new AssertionError("CriticHeuristic was not installed — params plumbing changed?");
        return modified;
    }

    @Override
    public NeuralBothMCTSPlayer copy() {
        NeuralBothMCTSPlayer c = new NeuralBothMCTSPlayer(
                (MCTSParams) getParameters().copy(), weightsPath, featureClass);
        if (getForwardModel() != null) c.setForwardModel(getForwardModel());
        c.setName(toString());
        return c;
    }
}
