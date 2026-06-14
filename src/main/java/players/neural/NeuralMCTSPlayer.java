package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * MCTS player using a trained PPO policy as the rollout strategy.
 *
 * Two constructors:
 *   1. NeuralMCTSPlayer(weights, features) — uses default MCTSParams. Back-compat.
 *   2. NeuralMCTSPlayer(baseParams, weights, features) — starts from baseParams
 *      (typically a tuned game-specific config loaded from JSON), copies it, and
 *      applies only the rollout-strategy override. All other tuning is preserved.
 *
 * The second constructor is what TunedTournament uses to ensure plain MCTSPlayer
 * and the neural variants differ ONLY in the rollout-strategy injection.
 */
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

    /**
     * Copies the base params and applies the neural rollout override.
     * Idempotent — re-applying the same overrides has no additional effect.
     */
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
