from espnet2.train.trainer import *
from espnet2.torch_utils.recursive_op import recursive_gather
from espnet2.sre.utils import calculate_eer
import numpy as np
class SRETrainer(Trainer):

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        scores = []
        labels = []
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            _, stats, weight = model(**batch)
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_gather()
                #gathered_stats1 = [torch.empty_like(stats['score']) for _ in range(torch.distributed.get_world_size())]
                #gathered_stats2 = [torch.empty_like(stats['label']) for _ in range(torch.distributed.get_world_size())]
                #torch.distributed.all_gather(gathered_stats1, stats['score'])
                #torch.distributed.all_gather(gathered_stats2, stats['label'])
                #stats = {'score': gathered_stats1, 'label': gathered_stats2}
                #gathered_stats = {'score': gathered_stats1, 'label': gathered_stats2}
                ws = torch.distributed.get_world_size()
                batch_sizes = [torch.tensor(1).to("cuda" if ngpu > 0 else "cpu") for _ in range(ws)]
                torch.distributed.all_gather(batch_sizes, torch.tensor(stats['score'].shape[0]).to("cuda" if ngpu > 0 else "cpu") )
                m_batch_size = max(batch_sizes)
                gathered_stats = {k: [torch.empty(m_batch_size, dtype=v.dtype).to("cuda" if ngpu > 0 else "cpu") for _ in range(ws)] for k, v in stats.items()}
                for k, v in stats.items():
                    if v.shape[0] < m_batch_size:
                        nans = torch.ones(m_batch_size - v.shape[0], dtype=v.dtype).to("cuda" if ngpu > 0 else "cpu") * -1
                        stats[k] = torch.cat([v, nans])

                stats = recursive_gather(gathered_stats, stats, distributed)
            scores.extend([score.cpu().view(-1) for score in stats['score']])
            labels.extend([label.cpu().view(-1) for label in stats['label']])
            # Note(Naohiro): Do not report score in each batch
            #                because EER must be calculated over all batches

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
        scores = torch.cat(scores)
        labels = torch.cat(labels)

        eer, threshold, fpr, tpr, auc = calculate_eer(labels, scores)
        reporter.register(dict(eer=eer, threshold=threshold, auc=auc))