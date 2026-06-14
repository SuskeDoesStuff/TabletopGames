package players.neural;

import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * MCTS player using the trained critic head as the leaf-value heuristic,
 * with rollouts disabled (rolloutLength = 0).
 *
 * IMPORTANT IMPLEMENTATION NOTE — why setParameterValue, not field assignment:
 * "heuristic" is a REGISTRY-BACKED tunable parameter in MCTSParams
 * (addTunableParameter("heuristic", IStateHeuristic.class, ...)), and
 * TunableParameters.setParameterValue ends with `if (resetOn) _reset()`,
 * which re-syncs ALL public fields from the registry — including
 * `heuristic = getParameterValue("heuristic")` (MCTSParams._reset, line ~178).
 *
 * A direct field write (`params.heuristic = critic`) is therefore silently
 * WIPED by the next setParameterValue call (and by params.copy(), which also
 * ends in _reset()). The critic must be installed via
 * setParameterValue("heuristic", critic) so it lives in the registry and
 * survives both _reset() and copy().
 */
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
        // Registry-backed: survives _reset() and copy(). Replaces whatever
        // heuristic the base config specified (e.g., WinOnlyHeuristic).
        modified.setParameterValue("heuristic", new CriticHeuristic(weights, features));
        // Disable rollouts entirely; critic evaluates the leaf directly.
        modified.setParameterValue("rolloutLength", 0);

        // Sanity check: fail loudly if the critic did not survive.
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
