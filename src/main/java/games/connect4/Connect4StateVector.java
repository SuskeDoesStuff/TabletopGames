package games.connect4;

import core.AbstractGameState;
import core.components.BoardNode;
import core.components.Token;
import core.interfaces.IStateFeatureVector;
import core.interfaces.IStateKey;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Per-cell board encoding of a Connect4 game state, perspective-relative.
 *
 * For each cell of the 8×8 grid in row-major order, emits:
 *   +1.0  if the cell holds the calling player's piece
 *    0.0  if the cell is empty
 *   -1.0  if the cell holds the opponent's piece
 *
 * Vector dimension: 64 (8 rows × 8 columns).
 *
 * If you change the Connect4 grid size away from 8×8 via Connect4GameParameters,
 * update the row/col ranges in `names` to match. The doubleVector method already
 * uses `state.gridBoard.flattenGrid()` and therefore reads the actual grid size
 * at runtime, but `names()` is fixed at class construction and must agree.
 */
public class Connect4StateVector implements IStateFeatureVector, IStateKey {

    // 8 rows × 8 columns = 64 feature names ("row:col" indexed).
    // Previously this was IntStream.range(0, 3) for columns — a bug that made
    // names().length = 24 while doubleVector returned 64. Fixed to 0..8.
    private final String[] names = (String[]) IntStream.range(0, 8).boxed().flatMap(row ->
            IntStream.range(0, 8).mapToObj(col -> String.format("%d:%d", row, col))
    ).toArray(String[]::new);

    @Override
    public double[] doubleVector(AbstractGameState gs, int playerID) {
        Connect4GameState state = (Connect4GameState) gs;
        String playerChar = Connect4Constants.playerMapping.get(playerID).getComponentName();

        return Arrays.stream(state.gridBoard.flattenGrid()).mapToDouble(c -> {
            String pos = c.getComponentName();
            if (pos.equals(playerChar)) {
                return 1.0;
            } else if (pos.equals(Connect4Constants.emptyCell)) {
                return 0.0;
            } else { // opponent's piece
                return -1.0;
            }
        }).toArray();
    }

    @Override
    public String[] names() {
        return names;
    }
}
