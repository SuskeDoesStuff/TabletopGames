package players.neural;

import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

// MCTS agent with no rollouts and critic head as the leaf evaluator
public class NeuralCriticMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralCriticMCTSPlayer(String weightsPath, String featureClass) {
        this(new MCTSParams(), weightsPath, featureClass);
    }

    public NeuralCriticMCTSPlayer(MCTSParams baseParams, String weightsPath, String featureClass) {
        super(applyCritic(baseParams, weightsPath, featureClass), "MCTS-CriticHeuristic");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams applyCritic(MCTSParams base, String weights, String features) {
        MCTSParams modified = (MCTSParams) base.copy();

        modified.setParameterValue("heuristic", new CriticHeuristic(weights, features));
        modified.setParameterValue("rolloutLength", 0);

        if (!(modified.heuristic instanceof CriticHeuristic))
            throw new AssertionError("CriticHeuristic was not installed — params plumbing changed?");
        return modified;
    }

    @Override
    public NeuralCriticMCTSPlayer copy() {
        NeuralCriticMCTSPlayer c = new NeuralCriticMCTSPlayer(
                (MCTSParams) getParameters().copy(), weightsPath, featureClass);
        if (getForwardModel() != null) c.setForwardModel(getForwardModel());
        c.setName(toString());
        return c;
    }
}
