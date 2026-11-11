## LoRa
*Slide "theory 3 LLM"*
C'è davvero bisogno di aggiustare tutti i parametri di un LLM quando faccio FT? Certe volte infatti un modello contiene più
parametri di quelli richiesti. Questo è però utile in fase di pretraining, perché diventa bravo a predire il prossimo token.

> A questo proposito, quando ho già imparato una conoscenza generica in pre-training, durante il fine-tuning (e poi in produzione) non mi servirà usarli tutti.

LoRA non tocca tutti i parametri del modello, ne aggiusta alcuni aggiungendo delle **matrici di cambiamento**. In questo caso 
i parametri della matrice di cambiamento vengono a loro volta *imparati* per capire *quale è il modo migliore di cambiare pesi* nel modello originale.

Il fine-tuning, genera gli *adapter*, toggle finale che può essere aggiunto ad un modello base senza dover avere più copie dei modelli. 

QLoRA è la verisone quantizzata di LoRA, fa quantizzazione a 4 bit risparmiando ancora memoria. Successivamente si può usare anche **Unsloth** per velocizzare ancora la cosa. 
