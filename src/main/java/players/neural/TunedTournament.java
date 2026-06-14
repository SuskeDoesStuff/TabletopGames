package players.neural;

import core.AbstractPlayer;
import evaluation.RunArg;
import evaluation.listeners.MetricsGameListener;
import evaluation.metrics.AbstractMetric;
import evaluation.metrics.GameMetrics;
import evaluation.metrics.IDataLogger;
import evaluation.tournaments.RoundRobinTournament;
import games.GameType;
import players.PlayerConstants;
import players.mcts.MCTSMetrics;
import players.mcts.MCTSParams;
import players.mcts.MCTSPlayer;
import players.simple.RandomPlayer;
import utilities.JSONUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * Tournament runner using a tuned MCTSParams config (loaded from JSON) as the
 * shared base for all MCTS-family agents, with the neural variants layering
 * their specific overrides on top.
 *
 * All MCTS agents — plain MCTSPlayer and the three neural variants — start
 * from the same tuned base (treePolicy, FPU, reuseTree, etc.). They differ
 * ONLY in:
 *   - NeuralMCTSPlayer        : neural rollout strategy
 *   - NeuralCriticMCTSPlayer  : critic heuristic + rolloutLength=0
 *   - NeuralBothMCTSPlayer    : both, with rolloutLength=5
 *
 * Output:
 *   - Per-event CSVs (via MetricsGameListener) in the outputDir.
 *   - A `tournament.txt` file in the same outputDir capturing all stdout/stderr
 *     including the final ranking, head-to-head stats, and config printout.
 *
 * Usage:
 *   java -cp target/TAG.jar players.neural.TunedTournament \
 *       <game> <paramsJson> <weights> <features> \
 *       [budgetType] [budget] [matchups] [nPlayers] [outputDir]
 *
 * Example:
 *   java -cp target/TAG.jar players.neural.TunedTournament \
 *       Connect4 json/players/gameSpecific/Connect4.json \
 *       /home/suske/projects/RL-MCTS/PyTAG/connect4_weights.txt \
 *       games.connect4.Connect4StateVector \
 *       BUDGET_TIME 100 3000 2
 */
public class TunedTournament {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            printUsage();
            System.exit(1);
        }

        GameType game = GameType.valueOf(args[0]);
        String paramsPath = args[1];
        String weightsPath = args[2];
        String featureClass = args[3];
        PlayerConstants budgetType = args.length > 4
                ? PlayerConstants.valueOf(args[4]) : PlayerConstants.BUDGET_TIME;
        int budget = args.length > 5 ? Integer.parseInt(args[5]) : 100;
        int matchups = args.length > 6 ? Integer.parseInt(args[6]) : 3000;
        int nPlayers = args.length > 7 ? Integer.parseInt(args[7]) : 2;
        String outDir = args.length > 8 ? args[8]
                : "metrics/tuned_" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());

        // Ensure the output directory exists, then tee stdout/stderr to a
        // tournament.txt inside it. From this point on every println goes to
        // both the terminal AND the file.
        new File(outDir).mkdirs();
        File logFile = new File(outDir, "tournament.txt");
        PrintStream consoleStdout = System.out;
        PrintStream consoleStderr = System.err;
        FileOutputStream logFos = new FileOutputStream(logFile, false);
        PrintStream teeOut = new PrintStream(new TeeOutputStream(consoleStdout, logFos), true);
        PrintStream teeErr = new PrintStream(new TeeOutputStream(consoleStderr, logFos), true);
        System.setOut(teeOut);
        System.setErr(teeErr);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            teeOut.flush(); teeErr.flush();
            try { logFos.flush(); logFos.close(); } catch (IOException ignored) {}
        }));

        System.out.println("=== TunedTournament configuration ===");
        System.out.println("  Game:        " + game);
        System.out.println("  Tuned JSON:  " + paramsPath);
        System.out.println("  Weights:     " + weightsPath);
        System.out.println("  Features:    " + featureClass);
        System.out.println("  BudgetType:  " + budgetType);
        System.out.println("  Budget:      " + budget +
                (budgetType == PlayerConstants.BUDGET_TIME ? " ms" : " (count)"));
        System.out.println("  Matchups:    " + matchups);
        System.out.println("  N players:   " + nPlayers);
        System.out.println("  Output dir:  " + outDir);
        System.out.println("  Log file:    " + logFile.getAbsolutePath());
        System.out.println();

        // Load the tuned config
        MCTSParams tunedBase = JSONUtils.loadClassFromFile(paramsPath);
        // Override budget settings (JSON typically has -999 as a placeholder).
        // MUST go through setParameterValue: budget/budgetType are registry-backed
        // tunable parameters, and params.copy() ends in _reset(), which restores
        // public fields FROM the registry. A direct field write here would be
        // silently reverted to the JSON's values (-999!) in every agent's copy.
        tunedBase.setParameterValue("budgetType", budgetType);
        tunedBase.setParameterValue("budget", budget);

        System.out.println("Tuned config loaded; key fields:");
        System.out.println("  class           = " + tunedBase.getClass().getSimpleName());
        System.out.println("  treePolicy      = " + tunedBase.treePolicy);
        System.out.println("  reuseTree       = " + tunedBase.reuseTree);
        System.out.println("  information     = " + tunedBase.information);
        System.out.println("  firstPlayUrgency= " + tunedBase.firstPlayUrgency);
        System.out.println("  rolloutLength   = " + tunedBase.rolloutLength);
        System.out.println("  maxTreeDepth    = " + tunedBase.maxTreeDepth);
        System.out.println("  heuristic       = " +
                (tunedBase.heuristic != null ? tunedBase.heuristic.getClass().getSimpleName() : "null"));
        System.out.println("  budgetType      = " + tunedBase.budgetType);
        System.out.println("  budget          = " + tunedBase.budget);
        System.out.println();

        // Build the agent list — each gets its own copy of the tuned base
        List<AbstractPlayer> agents = new ArrayList<>();
        MCTSPlayer plainMCTS = new MCTSPlayer((MCTSParams) tunedBase.copy(), "MCTSPlayer");
        // Verify the budget override survived the copy (it would not have, if set
        // via direct field assignment — see comment above).
        MCTSParams check = (MCTSParams) plainMCTS.getParameters();
        if (check.budget != budget || check.budgetType != budgetType)
            throw new AssertionError("Budget override lost in copy: budget=" + check.budget
                    + " budgetType=" + check.budgetType);
        agents.add(plainMCTS);
        agents.add(new NeuralMCTSPlayer((MCTSParams) tunedBase.copy(), weightsPath, featureClass));
        agents.add(new NeuralCriticMCTSPlayer((MCTSParams) tunedBase.copy(), weightsPath, featureClass));
        agents.add(new NeuralBothMCTSPlayer((MCTSParams) tunedBase.copy(), weightsPath, featureClass));
        agents.add(new NeuralRolloutPlayer(weightsPath, featureClass));
        agents.add(new RandomPlayer());

        // ---- Print the full player roster (all params) BEFORE the run ----
        // Format:  PlayerName
        //              param: value
        //              ...
        System.out.println("=== Player roster (effective parameters) ===");
        for (AbstractPlayer a : agents) {
            System.out.println(players.neural.ParamFormat.block(a, "    "));
            System.out.println();
        }
        System.out.println("=== Begin tournament ===");
        System.out.println();

        Map<RunArg, Object> config = RunArg.parseConfig(new String[]{
                "mode=exhaustive",
                "matchups=" + matchups,
                "verbose=true",
                "reportPeriod=" + matchups,
        }, Arrays.asList(RunArg.Usage.RunGames));

        RoundRobinTournament tournament = new RoundRobinTournament(
                agents, game, nPlayers, null, config);

        // Metric listener:
        //  - ComputationTimes  → per-decision wall-clock per agent (CRITICAL for fair-compute analysis)
        //  - Winner, FinalScore, OrdinalPosition, GameScore, Decisions → outcome data
        //  - TreeStats         → MCTS-specific tree-size/depth stats
        AbstractMetric[] metrics = new AbstractMetric[]{
                new GameMetrics.ComputationTimes(),
                new GameMetrics.Winner(),
                new GameMetrics.FinalScore(),
                new GameMetrics.OrdinalPosition(),
                new GameMetrics.GameScore(),
                new GameMetrics.Decisions(),
                new MCTSMetrics.TreeStats(),
        };
        MetricsGameListener listener = new MetricsGameListener(
                IDataLogger.ReportDestination.ToFile,
                new IDataLogger.ReportType[]{
                        IDataLogger.ReportType.RawDataPerEvent,
                        IDataLogger.ReportType.Summary,
                },
                metrics);
        listener.setOutputDirectory(outDir);
        tournament.addListener(listener);

        // Per-game parameter logging: prints both players' effective parameter
        // registries before every match (captured in tournament.txt via the tee).
        tournament.addListener(new GameParamsLogger());

        tournament.run();

        System.out.println();
        System.out.println("=== Tournament complete ===");
        System.out.println("Metrics written to: " + outDir + "/");
        System.out.println();
        System.out.println("Key files to inspect:");
        System.out.println("  " + outDir + "/tournament.txt       -> full console log (this run)");
        System.out.println("  " + outDir + "/ComputationTimes*    -> per-decision wall-clock per agent");
        System.out.println("  " + outDir + "/Winner*              -> game outcomes");
        System.out.println("  " + outDir + "/OrdinalPosition*     -> final rankings");
        System.out.println("  " + outDir + "/TreeStats*           -> MCTS tree statistics");

        // Flush so everything reaches the file before JVM exit
        System.out.flush();
        System.err.flush();
    }

    private static void printUsage() {
        System.err.println("Usage: TunedTournament <game> <paramsJson> <weights> <features> " +
                "[budgetType] [budget] [matchups] [nPlayers] [outputDir]");
        System.err.println();
        System.err.println("Required:");
        System.err.println("  game        - GameType enum, e.g., Connect4 or Diamant");
        System.err.println("  paramsJson  - path to tuned MCTSParams JSON file");
        System.err.println("                e.g., json/players/gameSpecific/Connect4.json");
        System.err.println("  weights     - path to neural-network weights file");
        System.err.println("  features    - feature class, e.g., games.connect4.Connect4StateVector");
        System.err.println();
        System.err.println("Optional (defaults in parentheses):");
        System.err.println("  budgetType  - BUDGET_TIME | BUDGET_FM_CALLS | BUDGET_ITERATIONS |");
        System.err.println("                BUDGET_COPY_CALLS | BUDGET_FMANDCOPY_CALLS  (BUDGET_TIME)");
        System.err.println("  budget      - int budget value  (100)");
        System.err.println("                For BUDGET_TIME this is milliseconds.");
        System.err.println("                For BUDGET_FM_CALLS this is a count.");
        System.err.println("  matchups    - total games  (3000)");
        System.err.println("  nPlayers    - seats per game  (2)");
        System.err.println("  outputDir   - where to write metric CSVs and tournament.txt log");
        System.err.println("                (metrics/tuned_<timestamp>/)");
    }

    /**
     * OutputStream wrapper that fans every write to two downstream streams.
     * Used to mirror stdout/stderr into a file while keeping terminal output.
     */
    private static class TeeOutputStream extends OutputStream {
        private final OutputStream a, b;
        TeeOutputStream(OutputStream a, OutputStream b) { this.a = a; this.b = b; }
        @Override public void write(int x) throws IOException {
            a.write(x); b.write(x);
        }
        @Override public void write(byte[] buf, int off, int len) throws IOException {
            a.write(buf, off, len); b.write(buf, off, len);
        }
        @Override public void flush() throws IOException { a.flush(); b.flush(); }
        @Override public void close() throws IOException {
            try { a.close(); } finally { b.close(); }
        }
    }
}
