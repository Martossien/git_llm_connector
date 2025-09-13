========================================================
GUIDE ANTI-PIÈGES — Open WebUI Tools (Lessons Learned)
========================================================

TL;DR
-----
1) Méthodes publiques simples (pas de **kwargs, pas de paramètres exotiques).
2) __init__ ultra-léger (pas d’I/O lourde ni de logique fragile).
3) Valves : toujours une classe Pydantic valide OU pas de valves du tout.
4) Schéma de fonctions : noms/annotations propres → JSON Schema propre.
5) Logging partout, y compris auto-vérification du schéma au chargement.

Détails (avec symptômes, cause, solution)
-----------------------------------------

A. Paramètres “magiques” (__event_emitter__, __user__, etc.)
------------------------------------------------------------
• Symptôme :
  - 400 sur /api/chat/completions ou warnings Pydantic
    “fields may not start with an underscore, ignoring '__event_emitter__'”.

• Cause :
  - Le loader convertit la signature Python → modèle Pydantic.
    Les noms commençant par “__” ne sont pas des champs valides pour le schéma.
    Selon la version d’Open WebUI, ils peuvent être ignorés (warning) ou
    provoquer un schéma ambigu.

• À NE PAS FAIRE :
  - Exposer des méthodes publiques avec des paramètres commençant par “__”.
  - Mélanger des méthodes statiques/sans ces params avec du code qui en
    dépend “magiquement”.

• À FAIRE :
  - Méthodes publiques sans ces paramètres spéciaux.
  - Si besoin d’un event emitter, gérer-le en interne (stockage optionnel,
    injection hors signature publique) ou via une méthode interne non exposée.


B. **kwargs / signatures non déclaratives
-----------------------------------------
• Symptôme :
  - 400 “type” ou erreurs de conversion de schéma (les params deviennent
    impossibles à décrire en JSON Schema).

• Cause :
  - Le générateur de schéma d’Open WebUI doit connaître précisément
    les paramètres. **kwargs/args posent problème.

• À NE PAS FAIRE :
  - Déclarer des outils avec **kwargs / *args en public.

• À FAIRE :
  - Signatures strictes, types simples : str, int, bool, float.
  - Valeurs par défaut raisonnables, pas d’union exotique.


C. Classe Valves absente / invalide
-----------------------------------
• Symptôme :
  - 500 sur /api/v1/tools/id/<id>/valves/spec
    (ex. “AttributeError: 'NoneType' object has no attribute 'schema'”).

• Cause :
  - Le backend s’attend à `Valves` (Pydantic) pour exposer le schéma.
    Si `Valves` est None, ou non-Pydantic, ou contient des champs non
    sérialisables → plantage du routeur.

• À NE PAS FAIRE :
  - Définir Valves=None ou comme objet arbitraire.

• À FAIRE :
  - Soit aucune valves → ne pas exposer de “Valves” du tout.
  - Soit une Valves **Pydantic BaseModel** minimaliste (types simples).
  - Pas de propriétés calculées, pas d’objets Path, pas de Callable, etc.


D. Nom de paramètre “piège” (ex. “type”) et schémas ambigus
-----------------------------------------------------------
• Symptôme :
  - 400 “type” côté /api/chat/completions (erreur opaque).

• Cause :
  - Le JSON Schema des outils exige “type: 'object'” avec “properties”.
    Des collisions sémantiques (param nommé “type”, annotations bizarres,
    conversions non prévues) peuvent produire un schéma invalide.

• À NE PAS FAIRE :
  - Appeler un paramètre “type”, “properties”, “items”, etc. (mots clés JSON
    Schema), surtout si l’annotation n’est pas triviale.

• À FAIRE :
  - Préférer des noms clairs : `mode`, `kind`, `variant`, `section`, etc.
  - Conserver des annotations simples (str/bool/int/float).


E. __init__ trop lourd (I/O, réseau, side effects)
--------------------------------------------------
• Symptôme :
  - Pas de logs d’exécution de tool en conversation, mais logs d’import OK.
  - Comportements erratiques dès le premier appel (400).

• Cause :
  - Au chargement du module, __init__ exécute des I/O (création répertoires,
    accès réseau, lecture de fichiers) → exceptions silencieuses, état partiel,
    ou délais qui se combinent mal avec l’initialisation d’Open WebUI.

• À NE PAS FAIRE :
  - Cloner des dépôts, sonder des binaires, lire des gros fichiers dans __init__.

• À FAIRE :
  - __init__ minimal : préparer des chemins, configurer le logger, c’est tout.
  - Reporter toute I/O réelle à l’appel utilisateur (lazy).


F. Chemins “~” non fiables / résolution utilisateur
---------------------------------------------------
• Symptôme :
  - Fichiers/logs introuvables ou créés ailleurs que prévu.

• Cause :
  - “~” peut ne pas être celui que tu crois (service, env conda, etc.).

• À NE PAS FAIRE :
  - Forcer “~/xxx” si l’environnement ne l’expanse pas correctement.

• À FAIRE :
  - Utiliser `os.path.expanduser("~")` pour construire `/home/<user>/...`.
  - Logguer les chemins finaux dès l’init.


G. Mélanger CLI/model non compatibles
-------------------------------------
• Symptôme :
  - Lancement de `qwen` avec `--model gemini-...` (ou inversement), sorties vides.

• Cause :
  - Les CLIs n’acceptent pas toutes `--model` ou le modèle n’a pas de sens
    pour la CLI choisie.

• À NE PAS FAIRE :
  - Forcer systématiquement `--model`.

• À FAIRE :
  - Heuristique : n’ajouter `--model` que si le binaire et le nom du modèle
    “matchent” (ex. binaire “gemini” + modèle qui commence par “gemini”).
  - Sinon, omettre proprement `--model` et logguer un warning.


H. Logs insuffisants (ou trop verbeux au mauvais endroit)
---------------------------------------------------------
• Symptôme :
  - Difficile de savoir si l’échec vient du schéma, des valves, ou de la CLI.

• Cause :
  - Pas de “self-check” de schéma, logs uniquement côté opérations lourdes.

• À FAIRE :
  - Logger au chargement :
    - Version du tool, chemins résolus.
    - Liste des méthodes exposées + leurs signatures (sans les “__”).
    - “Schema probe” : vérifier que chaque méthode génère un schéma type=object
      et afficher les `properties` détectées.
  - Logger côté LLM :
    - binaire choisi, modèle effectif (ou “omitted”), taille du contexte, durée.
  - Toujours tronquer les sorties d’erreurs longues (ex. 200 chars).


I. Sécurité / robustesse du contexte
------------------------------------
• Symptôme :
  - Contexte énorme, secrets dans les logs, mémoire excessive.

• À NE PAS FAIRE :
  - Lire des fichiers trop gros en entier, pousser le contexte entier dans les logs.

• À FAIRE :
  - Limites strictes : MAX_CONTEXT_BYTES, MAX_BYTES_PER_FILE, filtrage par globs.
  - Redaction basique (masquer clés API, JWT, etc.) avant d’envoyer au LLM.
  - Marqueurs “[... CONTEXTE TRONQUÉ ...]” pour transparence.


J. Compatibilité ascendante (ne pas patcher le cœur)
----------------------------------------------------
• Rappel :
  - Éviter tout patch du code d’Open WebUI pour “faire marcher” un tool.
    Sinon, tu brises la compatibilité future.

• À FAIRE :
  - S’aligner sur le contrat minimal : signatures propres + schéma propre.
  - Détecter les capacités côté tool (ex. pas de valves ⇒ ne pas exposer /valves).


Patron minimal recommandé
-------------------------
class Tools:
    def __init__(self):
        # init léger : chemins + logger
        # pas d’I/O lourde ici
        pass

    # —— méthodes publiques : signatures simples, types simples ——
    def tool_health(self, dummy: str = "") -> str:
        return "ok"

    def my_action(self, repo: str, mode: str = "standard") -> str:
        try:
            # logique réelle (I/O) ici, pas dans __init__
            return "done"
        except Exception as e:
            # logger.exception(...) recommandé
            return f"❌ my_action: {e}"

    # —— helpers internes (non exposés) ——
    def _internal_helper(self, ...):
        pass


Checklist avant publication
---------------------------
[ ] Aucune méthode publique avec __param__ spécial, **kwargs, *args
[ ] Pas de paramètre nommé “type” (ou autre mot-clé JSON Schema)
[ ] Valves : soit une classe Pydantic simple, soit aucune valves (mais pas “None”)
[ ] __init__ léger, pas d’I/O
[ ] Logs : version, chemins, signatures exposées, probe schéma OK
[ ] LLM : timeout suffisant (p.ex. 900s), --model conditionnel
[ ] Contexte : limites strictes, redaction, tronquage
[ ] Noms/annotations simples → JSON Schema propre
[ ] Test d’un appel “Hello” SANS tools (sanity check)
[ ] Test d’un appel AVEC tool (1 méthode simple) puis complexifier incrémentalement

Fin — En appliquant ces règles, on évite :
- 400 “type” (schéma invalide),
- 400 au premier appel (signature/params ambigus),
- 500 sur /valves/spec (Valves None/invalides),
- comportements imprévisibles dus à __init__ lourd.

