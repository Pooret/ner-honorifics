def train(epoch):
  tr_loss, tr_accuracy = 0, 0
  nb_tr_examples, nb_tr_steps = 0, 0
  tr_preds, tr_labels = [], []

  # put model into training mode
  model.train()

  for idx, batch in enumerate(training_loader):

    ids = batch['ids'].to(device, dtype = torch.long)
    mask = batch['mask'].to(device, dtype = torch.bool)
    targets = batch['targets'].to(device, dtype = torch.long)

    loss_logits = model(input_ids = ids, attention_mask = mask, labels = targets)
    loss = loss_logits[0]
    tr_logits = loss_logits[1]
    tr_loss += loss.item()

    nb_tr_steps += 1
    nb_tr_examples += targets.size(0)

    # for every 100
    if idx % 100 == 0:
      loss_step = tr_loss / nb_tr_steps
      print(f"Training loss per 100 training steps: {loss_step}")

    # compute performance accuracy
    flattened_targets = targets.view(-1) # shape (batch_size * seq_len)
    active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_preds = torch.argmax(active_logits, axis=1)

    active_accuracy = mask.view(-1)
    targets = torch.masked_select(flattened_targets, active_accuracy)
    preds = torch.masked_select(flattened_preds, active_accuracy)

    tr_preds.extend(preds)
    tr_labels.extend(targets)

    tmp_tr_acccuracy = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
    tr_accuracy += tmp_tr_acccuracy

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(), max_norm = MAX_GRAD_NORM
    )

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # get and print loss/accuracy  
  epoch_loss = tr_loss / nb_tr_steps
  tr_accuracy = tr_accuracy / nb_tr_steps
  print(f"Training loss epoch: {epoch_loss}")
  print(f"Training accuracy epoch: {tr_accuracy}")