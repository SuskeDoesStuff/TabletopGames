package players.neural;

import core.AbstractPlayer;
import evaluation.RunArg;
import evaluation.tournaments.RoundRobinTournament;
import games.GameType;
import players.basicMCTS.BasicMCTSPlayer;
import players.mcts.MCTSPlayer;
import players.simple.RandomPlayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Stripped-down baseline tournament for investigating MCTS-variant differences
 * without any neural agents in the mix.
 *
 * Three agents only:
 *   1. MCTSPlayer       (the configurable MCTS in TAG)
 *   2. BasicMCTSPlayer  (a simpler MCTS implementation in TAG)
 *   3. RandomPlayer     (uniform-random baseline)
 *
 * Usage:
 *   java -cp target/TAG.jar players.neural.BaselineTournament <GameType> [matchups] [nPlayers]
 *
 * Example:
 *   java -cp target/TAG.jar players.neural.BaselineTournament Connect4 1500 2
 */
public class BaselineTournament {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: BaselineTournament <GameType> [matchups] [nPlayers]");
            System.err.println("  GameType: e.g., Connect4, Diamant, TicTacToeGame");
            System.err.println("  matchups: total tournament games (default 1500)");
            System.err.println("  nPlayers: seats per game (default 2)");
            System.exit(1);
        }

        GameType game = GameType.valueOf(args[0]);
        int matchups = args.length > 1 ? Integer.parseInt(args[1]) : 1500;
        int nPlayers = args.length > 2 ? Integer.parseInt(args[2]) : 2;

        List<AbstractPlayer> agents = new ArrayList<>();
        agents.add(new MCTSPlayer());
        agents.add(new BasicMCTSPlayer());
        agents.add(new RandomPlayer());

        Map<RunArg, Object> config = RunArg.parseConfig(new String[]{
                "mode=exhaustive",
                "matchups=" + matchups,
                "verbose=true",
                "reportPeriod=" + matchups,
        }, Arrays.asList(RunArg.Usage.RunGames));

        RoundRobinTournament tournament = new RoundRobinTournament(
                agents, game, nPlayers, null, config);
        tournament.run();
    }
}
