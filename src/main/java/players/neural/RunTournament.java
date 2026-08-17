package players.neural;

import core.AbstractPlayer;
import evaluation.RunArg;
import evaluation.tournaments.RoundRobinTournament;
import games.GameType;
import players.basicMCTS.BasicMCTSPlayer;
import players.mcts.MCTSPlayer;
import players.simple.RandomPlayer;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

// Custom tournament runners
public class RunTournament {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("usage: RunTournament <GameType> <weights.txt> "
                    + "<featureExtractorClass> [matchups] [nPlayers]");
            return;
        }
        GameType game = GameType.valueOf(args[0]);
        String weights = args[1];
        String featureClass = args[2];
        int matchups = args.length > 3 ? Integer.parseInt(args[3]) : 300;
        int nPlayers = args.length > 4 ? Integer.parseInt(args[4]) : 2;
        int budgetPerMove = 1000;

        List<AbstractPlayer> agents = Arrays.asList(
        new NeuralMCTSPlayer(weights, featureClass),
        new MCTSPlayer(),
        new BasicMCTSPlayer(),
        new RandomPlayer(),
        new NeuralRolloutPlayer(weights, featureClass),
        new NeuralCriticMCTSPlayer(weights, featureClass),
        new NeuralBothMCTSPlayer(weights, featureClass)   // <-- add this line
        );

        Map<RunArg, Object> config = RunArg.parseConfig(
                new String[]{}, Collections.singletonList(RunArg.Usage.RunGames));
        config.put(RunArg.mode, "exhaustive");
        config.put(RunArg.matchups, matchups);
        config.put(RunArg.budget, budgetPerMove);
        config.put(RunArg.verbose, true);

        RoundRobinTournament tournament =
                new RoundRobinTournament(agents, game, nPlayers, null, config);
        tournament.run();
    }

    private static AbstractPlayer named(AbstractPlayer p, String name) {
        p.setName(name);
        return p;
    }
}
