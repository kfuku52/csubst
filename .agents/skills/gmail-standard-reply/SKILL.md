---
name: gmail-standard-reply
description: Create, update, or send Gmail replies that preserve the original Gmail thread and the standard quoted conversation. Use whenever the user asks to reply, draft a reply, or save a reply draft in Gmail; do not use for unrelated new messages.
---

# Gmail Standard Reply

Make Gmail replies behave like using Gmail's normal **Reply** button. Preserve both the message relationship and the visible quoted history; merely sending a new message with a matching subject is not sufficient.

## Reply workflow

1. Identify the intended Gmail thread and the latest inbound message being answered. Read the complete message body and enough thread context to avoid quoting or answering the wrong message.
2. Use Gmail's reply relationship when the available Gmail action supports it: keep the same thread, set the original message as the reply target, preserve the subject, and retain the relevant `In-Reply-To`/`References` relationship. Use the original `Reply-To` address when present.
3. Follow Gmail recipient semantics:
   - For an ordinary reply, address the sender or `Reply-To` address.
   - Use Reply all only when the user requests it or the existing context clearly establishes that intent.
   - Select the user's Gmail address or alias that received the original message when possible.
4. Write the new reply above the quoted conversation.
5. Include the latest received message as Gmail-style quoted history. Preserve any earlier history already nested in that message instead of rebuilding or duplicating the entire thread.
6. If a matching draft already exists in the thread, update it instead of creating a duplicate.
7. Save as a draft unless the user explicitly asks to send. An explicit send request authorizes sending only the reviewed reply in scope; do not send other drafts or follow-up messages.

## Gmail quote format

When the Gmail action accepts HTML, use Gmail-compatible quote markup beneath the new text:

```html
<br>
<div class="gmail_quote gmail_quote_container">
  <div dir="ltr" class="gmail_attr">[date/time], [sender] &lt;[address]&gt;:</div>
  <blockquote class="gmail_quote" style="margin:0 0 0 .8ex;border-left:1px #ccc solid;padding-left:1ex">
    [original rendered message body]
  </blockquote>
</div>
```

Escape untrusted header text and preserve the original message's safe rendered formatting. Do not reattach old attachments; leave them represented in the existing thread.

When only plain text is supported, include a standard attribution line followed by the original message with each line prefixed by `> `. Do not omit the quoted history silently.

If the Gmail connector cannot preserve the reply relationship or quoted content, use Gmail's native Reply UI in an available signed-in browser. If neither route can access the original message, stop and explain what is missing rather than fabricating the quote.

## Verification

After creating or updating the reply, inspect the saved draft when the available surface permits it and confirm:

- it belongs to the original thread;
- From, To/Cc, and subject are correct;
- the new reply appears above the standard quoted history;
- the quote contains the actual latest inbound message and is not duplicated;
- the requested state is correct: draft saved or message sent.

Report whether the reply was saved or sent. Mention any limitation if the result only approximates Gmail's native reply behavior.
