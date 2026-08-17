package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

// MCTS player using a trained PPO policy as the rollout strategy
public class NeuralMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralMCTSPlayer(String weightsPath, String featureClass) {
        this(new MCTSParams(), weightsPath, featureClass);
    }

    public NeuralMCTSPlayer(MCTSParams baseParams, String weightsPath, String featureClass) {
        super(applyNeuralRollout(baseParams, weightsPath, featureClass), "MCTS-NeuralRollout");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams applyNeuralRollout(MCTSParams base, String weights, String features) {
        MCTSParams modified = (MCTSParams) base.copy();
        modified.setParameterValue("rolloutType", MCTSEnums.Strategies.CLASS);
        modified.setParameterValue("rolloutClass",
                "{\"class\":\"players.neural.NeuralRolloutPlayer\",\"args\":[\""
                        + weights + "\",\"" + features + "\"]}");
        return modified;
    }

    @Override
    public NeuralMCTSPlayer copy() {
        NeuralMCTSPlayer c = new NeuralMCTSPlayer(
                (MCTSParams) getParameters().copy(), weightsPath, featureClass);
        if (getForwardModel() != null) c.setForwardModel(getForwardModel());
        c.setName(toString());
        return c;
    }
}
