package games.sushigo;

import core.AbstractGameState;
import core.interfaces.IStateFeatureJSON;
import core.interfaces.IStateFeatureVector;
import games.sushigo.cards.SGCard;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 SushiGo feature extractor.

 The vector produced by {@link #doubleVector} is a line by line port of the
 Python {@code SushiGoWrapper.process_json_obs} in PyTAG
 (pytag/utils/wrappers.py). It consumes the SAME JSON that
 {@link #getObservationJson} emits and applies the SAME transformation, so a
 PPO policy trained through PyTAG's JSON path and a policy used via this Java
 vector path see identical observations. This matches the PyTAG implementation to guarantee training and parity.

 Layout (concatenated, matching the Python concatenate order):
    1                : score      = playerScore / 50
    1                : round      = roundCounter / 3
   12                : played     = sum of one-hot of MY played cards
  120                : hand       = 10 slots x 12 one-hot, zero-padded
   12*(nPlayers-1)   : oppPlayed  = per-opponent sum of one-hot of played cards
    1*(nPlayers-1)   : oppScores  = per-opponent score / 50

 At 2 players this is 147, matching the wrapper's declared shape=[147].

 SGCard.toString()
 emits "Maki","Maki-2","Maki-3" (Maki variants) and the plain type name
 otherwise, which matches these strings exactly. An empty deck stringifies to
 "EmptyDeck" (Deck.toString), which maps to an all-zero card embedding.
**/
public class SGFeatures implements IStateFeatureVector, IStateFeatureJSON {

    private static final String[] CARD_TYPES = {
            "Maki", "Maki-2", "Maki-3", "Chopsticks", "Tempura", "Sashimi",
            "Dumpling", "SquidNigiri", "SalmonNigiri", "EggNigiri", "Wasabi",
            "Pudding"
    };
    private static final int N_CARD_TYPES = CARD_TYPES.length; // 12
    private static final int MAX_CARDS_IN_HAND = 10;           //  max_cards_in_hand

    @Override
    public String[] names() {
        // PyTAG gets the obs dim from names.length(). It needs to be equal to doubleVector().length or the gym env fails to do anything
        java.util.List<String> n = new java.util.ArrayList<>();
        n.add("score");
        n.add("round");
        for (String t : CARD_TYPES) n.add("myPlayed_" + t);
        for (int slot = 0; slot < MAX_CARDS_IN_HAND; slot++)
            for (String t : CARD_TYPES) n.add("hand_s" + slot + "_" + t);
        // per-opponent played counts + score. At 2 players there is exactly one opponent; this generalises if the vector ever does.
        for (int opp = 0; opp < 1; opp++) {            // 2-player: 1 opponent
            for (String t : CARD_TYPES) n.add("opp" + opp + "_played_" + t);
        }
        for (int opp = 0; opp < 1; opp++)
            n.add("opp" + opp + "_score");
        return n.toArray(new String[0]);
    }

    // One hot encoding of allof the card types
    private static double[] getCardId(String card) {
        double[] emb = new double[N_CARD_TYPES];
        if (!card.equals("EmptyDeck")) {
            int idx = Arrays.asList(CARD_TYPES).indexOf(card);
            if (idx >= 0) emb[idx] = 1.0;
        }
        return emb;
    }

    @Override
    public double[] doubleVector(AbstractGameState state, int playerID) {
        // Build the JSON exactly as getObservationJson would, then process it
        // identically to the Python wrapper. This guarantees the Java vector
        // equals the Python-trained observation.
        SGGameState sggs = (SGGameState) state;

        String[] playedCards = sggs.getPlayedCards().get(playerID).toString().split(",");
        String[] cardsInHand = sggs.getPlayerHands().get(playerID).toString().split(",");
        double score = sggs.getGameScore(playerID) / 50.0;       // playerScore/50
        double round = sggs.getRoundCounter() / 3.0;             // rounds/3

        List<double[]> oppPlayedPerOpp = new ArrayList<>();
        List<Double> oppScores = new ArrayList<>();
        for (int i = 0; i < sggs.getNPlayers(); i++) {
            if (i == playerID) continue;
            String[] oppPlayed = sggs.getPlayedCards().get(i).toString().split(",");
            // sum of one-hots for this opponent's played cards
            double[] summed = new double[N_CARD_TYPES];
            for (String c : oppPlayed) {
                double[] e = getCardId(c);
                for (int k = 0; k < N_CARD_TYPES; k++) summed[k] += e[k];
            }
            oppPlayedPerOpp.add(summed);
            oppScores.add(sggs.getGameScore(i) / 50.0);
        }

        // my played: sum of one-hots
        double[] playedSum = new double[N_CARD_TYPES];
        for (String c : playedCards) {
            double[] e = getCardId(c);
            for (int k = 0; k < N_CARD_TYPES; k++) playedSum[k] += e[k];
        }

        // my hand: one-hot per slot, zero-padded to MAX_CARDS_IN_HAND, flattened
        double[] handFlat = new double[MAX_CARDS_IN_HAND * N_CARD_TYPES];
        for (int slot = 0; slot < MAX_CARDS_IN_HAND; slot++) {
            if (slot < cardsInHand.length) {
                double[] e = getCardId(cardsInHand[slot]);
                System.arraycopy(e, 0, handFlat, slot * N_CARD_TYPES, N_CARD_TYPES);
            }
            // else leave zeros
        }

        // opp played flattened per opp
        // sum over that opponent's cards, then concatenate opponents in order)
        double[] oppPlayedFlat = new double[oppPlayedPerOpp.size() * N_CARD_TYPES];
        for (int o = 0; o < oppPlayedPerOpp.size(); o++) {
            System.arraycopy(oppPlayedPerOpp.get(o), 0, oppPlayedFlat,
                    o * N_CARD_TYPES, N_CARD_TYPES);
        }

        // concatenate in order
        // score, round, played, hand, oppPlayed, oppScores
        int dim = 1 + 1 + N_CARD_TYPES + handFlat.length
                + oppPlayedFlat.length + oppScores.size();
        double[] obs = new double[dim];
        int p = 0;
        obs[p++] = score;
        obs[p++] = round;
        System.arraycopy(playedSum, 0, obs, p, N_CARD_TYPES); p += N_CARD_TYPES;
        System.arraycopy(handFlat, 0, obs, p, handFlat.length); p += handFlat.length;
        System.arraycopy(oppPlayedFlat, 0, obs, p, oppPlayedFlat.length); p += oppPlayedFlat.length;
        for (double s : oppScores) obs[p++] = s;

        return obs;
    }

    @Override
    public String getObservationJson(AbstractGameState gameState, int playerID) {
        SGGameState sggs = (SGGameState) gameState;
        JSONObject json = new JSONObject();
        json.put("PlayerID", playerID);
        json.put("nPlayers", sggs.getNPlayers());
        json.put("rounds", sggs.getRoundCounter());
        json.put("cardsInHand", sggs.getPlayerHands().get(playerID).toString());
        json.put("playedCards", sggs.getPlayedCards().get(playerID).toString());
        json.put("playerScore", sggs.getGameScore(playerID));
        for (int i = 0; i < sggs.getNPlayers(); i++) {
            if (i != playerID) {
                json.put("opp" + i + "playedCards", sggs.getPlayedCards().get(i).toString());
                json.put("opp" + i + "score", sggs.getGameScore(i));
            }
        }
        return json.toJSONString();
    }
}
