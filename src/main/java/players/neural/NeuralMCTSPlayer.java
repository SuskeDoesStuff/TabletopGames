package players.neural;

import players.mcts.MCTSEnums;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;

/**
 * An MCTS player that uses a trained PyTAG policy as its rollout strategy.
 *
 * It IS a full MCTS agent (selection / expansion / rollout / backup); the only
 * change from the stock MCTSPlayer is that random playouts are replaced by the
 * neural NeuralRolloutPlayer. Everything else about the search is unchanged, so
 * comparing this against a plain MCTSPlayer isolates exactly one variable: the
 * rollout policy.
 *
 * Self-contained: the rollout policy is specified as an INLINE JSON spec on the
 * rolloutClass parameter, so there is no separate config file to manage and the
 * setting survives MCTS's internal parameter resets.
 *
 *   new NeuralMCTSPlayer("/abs/ttt_weights.txt", "games.tictactoe.TTTFeatures")
 *   new NeuralMCTSPlayer("/abs/diamant_weights.txt", "games.diamant.DiamantFeatures")
 *   new NeuralMCTSPlayer("/abs/sushigo_weights.txt", "games.sushigo.SGFeatures")
 */
public class NeuralMCTSPlayer extends MCTSPlayer {

    private final String weightsPath;
    private final String featureClass;

    public NeuralMCTSPlayer(String weightsPath, String featureClass) {
        super(buildParams(weightsPath, featureClass), "MCTS-NeuralRollout");
        this.weightsPath = weightsPath;
        this.featureClass = featureClass;
    }

    private static MCTSParams buildParams(String weightsPath, String featureClass) {
        MCTSParams params = new MCTSParams();
        // Inline class spec: loadClass() parses this directly (it contains '{'),
        // matching the NeuralRolloutPlayer(String, String) constructor via "args".
        // weightsPath must NOT end in .json (the loader special-cases that).
        String spec = "{\"class\":\"players.neural.NeuralRolloutPlayer\",\"args\":[\""
                + weightsPath + "\",\"" + featureClass + "\"]}";
        params.setParameterValue("rolloutType", MCTSEnums.Strategies.CLASS);
        params.setParameterValue("rolloutClass", spec);
        return params;
    }

    @Override
    public NeuralMCTSPlayer copy() {
        NeuralMCTSPlayer c = new NeuralMCTSPlayer(weightsPath, featureClass);
        c.setForwardModel(getForwardModel());
        if (getParameters() != null) c.setBudget(getParameters().budget);
        c.setName(toString());
        return c;
    }
}
