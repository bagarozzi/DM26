# Retrieval-Augmented Generation
I normali LLM apprendono dal dataset e quindi tutte le loro risposte hanno una rilevanza statistica ma non sappiamo con esattezza in che pesi è immagazzinata una certa informazione.

Ci possono poi anche essere delle allucinazioni dovute al fatto che diverse informazioni si sovrappongono nella rete. 
- Conflitto di contesto
- Conflitto di comprensione
- Conflitto di fatti: si contraddice la conoscenza vera
- Reverse lookups: invertire una domanda già nota non fa rispondere

E' infatti impossibile andare a **modificare in maniera selettiva i pesi** per aggiungere informazioni, in più **i fatti rari non sono ricordati** perché statisticamente appaiono poco ed alcune non appaiono affatto (informazioni aziendali o estremamente di nicchia).
Per finire, le informazioni che cambiano dopo il training non possono essere aggiornate e non si possono avere fonti dalle risposte degli LLM.

Con RAG la conoscenza è all'**esterno del modello**. Il modello è diviso in due parti:
- **Retriever**: sceglie i documenti più rilevanti
- **Reader**: legge i documenti e da la risposta

I modelli possono essere **più piccoli** perché non serve memorizzare i fatti ma saper fare *ragionamenti e capire il linguaggio*. Come chi lavora con le leggi, non memorizza ma consulta. 

> Ora c'è direttamente la memorizzazione dei dati tramite embedding in un database di vettori.

Il RAG strutturato usa dati strutturati come SQL o grafi, perché gli LLM sanno interrogare basi di dati strutturate. 


