package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * Combined neural injection: policy as rollout strategy AND critic as leaf
 * heuristic. Rollout length is INHERITED from the base/tuned config (not
 * hardcoded), so NeuralBoth and NeuralRollout run identical rollouts and differ
 * only in the leaf heuristic (CriticHeuristic vs the config's WinOnlyHeuristic).
 *
 * IMPORTANT IMPLEMENTATION NOTE — why setParameterValue for the heuristic:
 * "heuristic" is a registry-backed tunable in MCTSParams, and every
 * setParameterValue call ends in _reset(), which restores `heuristic` from
 * the registry. A direct field write (`params.heuristic = critic`) followed
 * by ANY setParameterValue call is silently wiped. The critic must be
 * installed via setParameterValue("heuristic", ...) to survive.
 */
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
        // Policy as rollout strategy (registry-backed params)
        modified.setParameterValue("rolloutType", MCTSEnums.Strategies.CLASS);
        modified.setParameterValue("rolloutClass",
                "{\"class\":\"players.neural.NeuralRolloutPlayer\",\"args\":[\""
                        + weights + "\",\"" + features + "\"]}");
        // Critic as leaf-value heuristic — registry-backed, survives _reset/copy
        modified.setParameterValue("heuristic", new CriticHeuristic(weights, features));
        // NOTE: rolloutLength is deliberately NOT overridden. NeuralBoth inherits
        // whatever the base/tuned config sets (identical to NeuralRollout), so the
        // two agents run the SAME rollouts and differ ONLY in the leaf heuristic:
        // WinOnlyHeuristic (NeuralRollout) vs CriticHeuristic (this one).
        // To get the short-rollout variant where the critic does the leaf work,
        // add: modified.setParameterValue("rolloutLength", 5);

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
