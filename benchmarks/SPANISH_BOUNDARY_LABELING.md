# Spanish Subtitle-Boundary Labeling Guide

## Labels

| Label | Meaning |
|---|---|
| **JOIN** | The right cue continues the same sentence or syntactic utterance begun in the left cue, and combining them does not cross a clear speaker turn. |
| **BREAK** | The right cue starts an independent sentence, independent utterance, explicit speaker turn, caption, or other protected structural boundary. |
| **AMBIGUOUS** | Both interpretations are reasonably defensible given source punctuation, translation quality, speaker identity, or discourse structure. |

## Core Rules

1. **Semantic relatedness alone is never enough for JOIN.** Two sentences on the same topic by the same speaker are still separate sentences → BREAK.

2. **A question and its answer are two utterances** → BREAK.

3. **Speaker turns are always BREAK**, even when the first speaker's utterance is clearly incomplete.

4. **JOIN only when the right cue completes the syntactic unit started in the left cue** and there is no speaker boundary.

5. **Do not label based on what SaT or the current policy predicts.** Label by reading the Spanish text.

---

## JOIN Examples

### Sentence continues across cues

```
Left:  No sabía que iba a ser
Right: tan difícil encontrar trabajo.
```
→ JOIN — "tan difícil encontrar trabajo" is a complement that cannot stand alone.

### Clause with conjunction

```
Left:  Llegamos tarde porque
Right: el tráfico estaba horrible.
```
→ JOIN — "porque" introduces a subordinate clause that completes the thought.

### Verb + complement split

```
Left:  Quiero decirte algo
Right: muy importante.
```
→ JOIN — "muy importante" modifies "algo" and is not an independent utterance.

### Prepositional phrase complement

```
Left:  Estaba sentado encima
Right: de la caja.
```
→ JOIN — "de la caja" completes the prepositional phrase "encima de."

---

## BREAK Examples

### Two independent sentences, same speaker

```
Left:  Hoy hace mucho calor.
Right: Mañana va a llover.
```
→ BREAK — each is a complete sentence even though the same person says both.

### Question + answer

```
Left:  ¿A qué hora llegaste?
Right: Como a las ocho.
```
→ BREAK — the question and answer are separate utterances.

### Speaker turn with dash

```
Left:  - ¿Dónde está Juan?
Right: - No lo sé.
```
→ BREAK — dialogue dash marks a speaker turn.

### Caption or sound description

```
Left:  Y entonces todo cambió.
Right: (Música dramática)
```
→ BREAK — parenthetical sound description is a structural boundary.

### Sentence + correction/afterthought

```
Left:  Trae el libro azul.
Right: Bueno, mejor el rojo.
```
→ BREAK — "bueno" introduces a new independent utterance.

### Short standalone response

```
Left:  ¿Vas a venir?
Right: Sí.
```
→ BREAK — "sí" is a complete utterance response.

### Independent subordinate clause following a complete sentence

```
Left:  Terminamos el proyecto.
Right: Aunque no quedó perfecto.
```
→ BREAK — left is a complete sentence; right begins a new independent clause.

---

## AMBIGUOUS Examples

### Missing or ambiguous punctuation

```
Left:  No sé si decirte esto
Right: pero creo que deberías saberlo
```
→ AMBIGUOUS — without terminal punctuation on the left, the status between a single sentence or two fragments is unclear.

### Short clause that could be either

```
Left:  Ya sé que no te gusta
Right: pero no hay otra opción
```
→ AMBIGUOUS — "pero no hay otra opción" could be the second half of a compound sentence (JOIN) or an independent utterance (BREAK).

### Translation artifact

```
Left:  Estábamos hablando de la reunión.
Right: Y de lo que pasó después.
```
→ AMBIGUOUS — "Y de lo que pasó después" could be a new sentence starting with "Y" (BREAK) or a continuation (JOIN). Both readings are valid.

### Mid-sentence inserted cue

```
Left:  Creía que habías terminado,
Right: pero veo que no.
```
→ AMBIGUOUS — the comma could be read as a pause within one sentence or as line-broken independent clauses.

### Ellipsis continuation

```
Left:  Bueno, no sé…
Right: Quizá deberíamos preguntar.
```
→ AMBIGUOUS — the ellipsis signals hesitation but the right cue could be a new thought or a continuation.

---

## Special Notes

- **Dialogue dashes** (—, -, –) at the start of a cue usually indicate a speaker change → BREAK.
- **Inverted punctuation** (¿, ¡) is a strong signal for sentence start → BREAK.
- **Short responses** (sí, no, claro, vale, bueno, ok) are typically BREAK.
- **Coordinating conjunctions** (y, e, o, pero, sino) do not automatically mean JOIN — examine if the right cue is a complete clause or a continuation.
- **Subordinate conjunctions** (que, porque, cuando, mientras, aunque, si) at the start of a right cue often suggest JOIN.
- **Timing gap** is not a label criterion — ignore it for labeling.
