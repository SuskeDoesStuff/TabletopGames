package players.neural;

import core.AbstractPlayer;
import evaluation.RunArg;
import evaluation.tournaments.RoundRobinTournament;
import games.GameType;
import players.basicMCTS.BasicMCTSPlayer;
import players.mcts.MCTSPlayer;
import players.simple.RandomPlayer;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BaselineTournament {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: BaselineTournament <GameType> [matchups] [nPlayers] [budget]");
            System.err.println("  matchups: total games (default 1500)");
            System.err.println("  nPlayers: seats per game (default 2)");
            System.err.println("  budget:   explicit budget for both MCTS agents");
            System.err.println("            (default: each agent's own default)");
            System.exit(1);
        }

        GameType game = GameType.valueOf(args[0]);
        int matchups = args.length > 1 ? Integer.parseInt(args[1]) : 1500;
        int nPlayers = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        int budget   = args.length > 3 ? Integer.parseInt(args[3]) : -1;

        MCTSPlayer mcts = new MCTSPlayer();
        BasicMCTSPlayer basic = new BasicMCTSPlayer();

        System.out.println("=== Parameter inspection (before any override) ===");
        printBudgetInfo("MCTSPlayer      ", mcts.getParameters());
        printBudgetInfo("BasicMCTSPlayer ", basic.getParameters());
        System.out.println();

        if (budget > 0) {
            System.out.println("=== Applying explicit override: budget=" + budget + " ===");
            applyBudgetOverride(mcts.getParameters(), budget);
            applyBudgetOverride(basic.getParameters(), budget);
            printBudgetInfo("MCTSPlayer      ", mcts.getParameters());
            printBudgetInfo("BasicMCTSPlayer ", basic.getParameters());
            System.out.println();
        } else {
            System.out.println("(No --budget override; each agent uses its own defaults)\n");
        }

        List<AbstractPlayer> agents = new ArrayList<>();
        agents.add(mcts);
        agents.add(basic);
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

    private static void printBudgetInfo(String label, Object params) {
        try {
            Field budgetField = params.getClass().getField("budget");
            Field budgetTypeField = params.getClass().getField("budgetType");
            System.out.printf("  %s class=%s, budget=%s, budgetType=%s%n",
                    label,
                    params.getClass().getSimpleName(),
                    budgetField.get(params),
                    budgetTypeField.get(params));
        } catch (NoSuchFieldException e) {
            System.out.printf("  %s class=%s, could not find 'budget' or 'budgetType' field%n",
                    label, params.getClass().getSimpleName());
        } catch (IllegalAccessException e) {
            System.out.printf("  %s field access error: %s%n", label, e.getMessage());
        }
    }

    private static void applyBudgetOverride(Object params, int newBudget) {
        try {
            params.getClass().getMethod("setParameterValue", String.class, Object.class)
                    .invoke(params, "budget", newBudget);
        } catch (Exception e) {
            System.err.println("Could not set 'budget' on " +
                    params.getClass().getSimpleName() + ": " + e.getMessage());
        }
        try {
            params.getClass().getMethod("setParameterValue", String.class, Object.class)
                    .invoke(params, "budgetType", "BUDGET_FM_CALLS");
        } catch (Exception e) {
            
        }
    }
}
