package players.neural;

import core.AbstractPlayer;
import core.Game;
import evaluation.listeners.IGameListener;
import evaluation.metrics.Event;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

// Lightweight ParamLogger for TunedTournament
public class GameParamsLogger implements IGameListener {

    private Game game;
    private int gameCounter = 0;
    private final Set<String> seen = new HashSet<>();

    @Override
    public void onEvent(Event event) {
        if (event.type != Event.GameEvent.ABOUT_TO_START) return;
        if (game == null) return;

        gameCounter++;
        List<AbstractPlayer> players = game.getPlayers();

        StringBuilder line = new StringBuilder();
        line.append("Game ").append(gameCounter).append(": ");
        for (int i = 0; i < players.size(); i++) {
            if (i > 0) line.append(" vs ");
            line.append(players.get(i).toString());
        }
        System.out.println(line);

        // Full param dump only for agents not seen before
        for (AbstractPlayer p : players) {
            if (seen.add(p.toString())) {
                System.out.println(ParamFormat.block(p, "    "));
            }
        }
    }

    @Override public void report() {}
    @Override public void setGame(Game game) { this.game = game; }
    @Override public Game getGame() { return game; }
}
