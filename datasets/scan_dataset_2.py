import torch
import os
import torch.nn.functional as F
import numpy as np

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from pathlib import Path

from utils.data import read_flow

from utils.log import getLogger

from .flow_pair_detectron import TrackDataNoSH, MutliframeTrackDataNoSH, Mutliframe5TrackDataNoSH
from .flow_pair_detectron import TrackData, MultiTrackData, Multi5TrackData
from .flow_pair_detectron import TrackDataset

logger = getLogger(__name__)

def build_metadata(basepath, img_res, sequences=None, flow_dirs=tuple(), flow_gaps=tuple(), flow_res=None, ano_paths=tuple(), ano_res=None, img_suffix='.jpg', flow_suffix='.flo', ano_suffix='.png'):
    dataset_dict = {}
    basepath = Path(basepath)
    flow_res = flow_res if flow_res is not None else img_res
    ano_res = ano_res if ano_res is not None else img_res
    sequences = sequences or sorted([p.name for p in (basepath / 'JPEGImages' / img_res).iterdir() if p.is_dir()])
    if len(sequences) == 0:
        print(f'No sequences found in {basepath}')
    for seq in sequences:
        imgs = sorted((basepath / 'JPEGImages' / img_res / seq).glob(f'*{img_suffix}'))
        if len(imgs) == 0:
            print(f'No images found for sequence {seq}')
        frames = {fid: {'rgb': str(f.relative_to(basepath))} for fid, f in enumerate(imgs)}
        for fid in frames:
            frame = frames[fid]
            fname_stem = Path(frame['rgb']).stem
            frame['fname'] = fname_stem
            
            for flow_dir in flow_dirs:
                frame[flow_dir] = {}
                for gap in flow_gaps:
                    fwd_gap = gap
                    bwd_gap = -gap
                    fwd_path = basepath / f'{flow_dir}{fwd_gap}' / flow_res / seq / f"{fname_stem}{flow_suffix}"
                    bwd_path = basepath / f'{flow_dir}{bwd_gap}' / flow_res / seq / f"{fname_stem}{flow_suffix}"
                    if fwd_path.exists():
                        frame[flow_dir][fwd_gap] = str(fwd_path.relative_to(basepath))
                    if bwd_path.exists():
                        frame[flow_dir][bwd_gap] = str(bwd_path.relative_to(basepath))
            for ano_dir in ano_paths:
                ano_path = basepath / ano_dir / ano_res / seq / f"{fname_stem}{ano_suffix}"
                if ano_path.exists():
                    frame[ano_dir] = str(ano_path.relative_to(basepath))
        dataset_dict[seq] = frames
    return dataset_dict


class AnoDatasetOldStyle:
    def __init__(self, metadata, basepath, rbg_transforms=None, flow_resolution=None, flow_clip=float('inf'), ano_types=('Annotations',)):
        self.basepath = basepath
        self.metadata = metadata
        self.ano_types= ano_types
        self.rgb_transforms = rbg_transforms or TF.to_tensor
        self.index = self.__build_dataset_iter_list__()
        # self.flow_resolution = flow_resolution or (1088, 1920)
        self.flow_resolution = flow_resolution
        self.flow_clip = flow_clip

        ano = ano_types[0]
        self.data_dir = (self.basepath, self.basepath / 'JPEGImages' / '480p', self.basepath / ano / '480p') # For some lookup compatibility
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        return self.build_item(self.index[idx])
    
    def __build_dataset_iter_list__(self):
        return self.build_ano_list(self.metadata, ano_types=self.ano_types)

    @staticmethod
    def build_ano_list(metadata, ano_types=('Annotations',)):
        ano_list = []
        seqs = sorted(list(metadata.keys()))
        for seq in seqs:
            fids = sorted(list(metadata[seq].keys()))
            for fid in fids:
                ano_type = ano_types[0]
                if ano_type in metadata[seq][fid]:
                    item = [fid, seq, metadata[seq][fid]['rgb'], metadata[seq][fid][ano_type]]
                    for other_ano in ano_types[1:]:
                        if other_ano in metadata[seq][fid]:
                            item.append(metadata[seq][fid][other_ano])
                    ano_list.append(item)
        return ano_list

    def load_rgb(self, rgb_path_offset):
        pil_img = Image.open(self.basepath / rgb_path_offset).convert('RGB')
        original_rgb = TF.to_tensor(pil_img) * 255.
        rgb = self.rgb_transforms(pil_img).clamp(0., 1.) * 255.
        return rgb, original_rgb
    
    def load_ano(self, ano_path_offset):
        pil_img = Image.open(self.basepath / ano_path_offset)
        original_ano = torch.from_numpy(np.array(pil_img)).long()
        if original_ano.ndim == 3:
            original_ano = original_ano[..., 0]
        if original_ano.max() >= 255:
            original_ano = (original_ano > 128).long()
        return original_ano
    
    def load_dotflo(self, flow_path_offset, res=None):
        if res is None:
            res = self.flow_resolution
        flo_path = str(self.basepath / flow_path_offset)
        flo = read_flow(flo_path, res, False)
        flo = torch.from_numpy(flo)
        flo = flo.clamp(-self.flow_clip, self.flow_clip)
        return flo
    
    def build_item(self, item):
        fid, seq, rgb_path_offset, ano_path_offset, *others = item
        data = {
            'frame_id': fid,
            'seq': seq,
            'category': ano_path_offset.split('/')[-2:],  # Wierd but how it was in the original code
        }
        ano = self.load_ano(ano_path_offset)
        data['sem_seg_ori'] = ano
        rgb, original_rgb = self.load_rgb(rgb_path_offset)
        data['sem_seg'] = F.interpolate(ano.view(1,1,*ano.shape[-2:]).float(), mode='nearest', size=rgb.shape[-2:])[0, 0].long()
        data['rgb'] = rgb
        data['original_rgb'] = F.interpolate(original_rgb[None], mode='bicubic', size=data['sem_seg_ori'].shape[-2:], align_corners=False).clip(0.,255.)[0]
        
        flow_keys = list(self.metadata[seq][fid]['Flows_gap'].keys())
        if len(flow_keys) > 0:
            flow_key = flow_keys[0]
            data['flow'] = self.load_dotflo(self.metadata[seq][fid]['Flows_gap'][flow_key])
        else:
            data['flow'] = torch.zeros((2, *self.flow_resolution))
        return [data]
    
class FlowDatasetOldStyle(AnoDatasetOldStyle):
    def __build_dataset_iter_list__(self):
        return self.build_flow_list(self.metadata, flow_dirs=('Flows_gap',))
    
    @staticmethod
    def build_flow_list(metadata, flow_dirs=('Flows_gap',), flow_gaps=(1, 2)):
        flow_list = []
        seqs = sorted(list(metadata.keys()))
        for seq in seqs:
            fids = sorted(list(metadata[seq].keys()))
            for flow_dir in flow_dirs:
                valid_fids = [fid for fid in fids if flow_dir in metadata[seq][fid]]
                for gap in flow_gaps:
                    fwd_gap = gap
                    bwd_gap = -gap
                    fwd_fids = [fid for fid in valid_fids if fwd_gap in metadata[seq][fid][flow_dir]]
                    bwd_fids = [fid for fid in valid_fids if bwd_gap in metadata[seq][fid][flow_dir]]
                    paired_fids = sorted(list(set(fwd_fids).intersection(bwd_fids)))
                    for fid in paired_fids:
                        item = [fid, seq, gap, metadata[seq][fid]['rgb'], metadata[seq][fid][flow_dir][fwd_gap], metadata[seq][fid][flow_dir][bwd_gap]]
                        flow_list.append(item)
        return flow_list
    
    def build_item(self, item):
        fid, seq, gap, rgb_path_offset, fwd_flow_path_offset, bwd_flow_path_offset = item
        data = {
            'frame_id': fid,
            'seq': seq
        }
        rgb, _ = self.load_rgb(rgb_path_offset)
        data['rgb'] = rgb
        data['flow'] = self.load_dotflo(fwd_flow_path_offset)
        ano_path = self.metadata[seq][fid].get('Annotations', None)
        if ano_path is not None:
            ano = self.load_ano(ano_path)
            data['sem_seg'] = F.interpolate(ano.view(1,1,*ano.shape[-2:]).float(), mode='nearest', size=rgb.shape[-2:])[0, 0].long()
        else:
            data['sem_seg'] = torch.zeros((data['flow'].shape[-2:]), dtype=torch.long)
        if data['flow'].shape[-2:] != data['rgb'].shape[-2:]:
            data['flow_big'] = data['flow']
            data['flow'] = self.load_dotflo(fwd_flow_path_offset, res=data['rgb'].shape[-2:])
        return [data]

class FlowTrackPairDataset(AnoDatasetOldStyle):
    def __init__(self, cfg, metadata, basepath, rbg_transforms=None, flow_resolution=None, flow_clip=float('inf'), load_tracks='co2-noshmem-multi', flow_gaps=(1, 2), ano_types=('Annotations',)):
        self.flow_gaps = flow_gaps
        super().__init__(metadata, basepath, rbg_transforms, flow_resolution, flow_clip, ano_types)
        self.cfg = cfg
        self.load_tracks = load_tracks
        if self.load_tracks.startswith('co2') or self.load_tracks.startswith('boost'):
            seqs = list(self.metadata.keys())
            frame_name_path = (self.basepath / self.metadata[seqs[0]][0]['rgb']).parent.parent
            if self.load_tracks.startswith('co2'):
                track_path =  self.basepath / 'Tracks' / 'cotrackerv2_rel_stride4_aux2'
                if 'BW' in self.load_tracks:
                    track_path = self.basepath / 'Tracks' / 'cotrackerBW_12288'
            elif self.load_tracks.startswith('boost'):
                track_path =  self.basepath / 'Tracks' / 'boosttapir_stride4'
            self.tracks = {}
            for seq in seqs:
                if 'noshmem' in self.load_tracks:
                    if 'multi5' in self.load_tracks:
                        data = Mutliframe5TrackDataNoSH(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                    elif 'multi' in self.load_tracks:
                        data = MutliframeTrackDataNoSH(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                    else:
                        data = TrackDataNoSH(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                
                else:
                    if 'multi5' in self.load_tracks:
                        data = Multi5TrackData(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                    elif 'multi' in self.load_tracks:
                        data = MultiTrackData(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                    else:
                        data = TrackData(track_path, seq, n=self.cfg.NTRACKS, step=4, threshold=self.cfg.TRACK_VIS_THRESH, sampling='random', frame_name_path=frame_name_path)
                dataset = TrackDataset(data, 
                                        ctx_len=self.cfg.CTX_LEN, 
                                        filt=self.cfg.FILTERN, 
                                        padmode='constant' if 'zeropad' in self.load_tracks else None,
                                        scale=1,
                                        shift=0,
                                        ctx_mode=self.cfg.CTX_MODE)
                self.tracks[seq] = dataset
                logger.info(f'Loaded and indexed <{seq}> tracks : {data.T.shape} {data.f}')

    def __build_dataset_iter_list__(self):
        return self.build_flow_pair_list(self.metadata, flow_gaps=self.flow_gaps)
     
    @staticmethod
    def build_flow_pair_list(metadata, flow_dirs=('Flows_gap',), flow_gaps=(1, 2)):
        flow_list = []
        seqs = sorted(list(metadata.keys()))
        for seq in seqs:
            fids = sorted(list(metadata[seq].keys()))
            for flow_dir in flow_dirs:
                valid_fids = [fid for fid in fids if flow_dir in metadata[seq][fid]]
                for gap in flow_gaps:
                    fwd_gap = gap
                    bwd_gap = -gap
                    fwd_fids = [fid for fid in valid_fids if fwd_gap in metadata[seq][fid][flow_dir]]
                    bwd_fids = [fid-gap for fid in valid_fids if bwd_gap in metadata[seq][fid][flow_dir]]
                    fwdbwd_fids = sorted(list(set(fwd_fids).intersection(bwd_fids)))
                    for fid in fwdbwd_fids:
                        item = [fid, fid+gap, 
                                seq, gap, 
                                metadata[seq][fid]['rgb'], metadata[seq][fid+gap]['rgb'],
                                metadata[seq][fid][flow_dir][fwd_gap], metadata[seq][fid+gap][flow_dir][bwd_gap]]
                        flow_list.append(item)
        return flow_list
    
    def load_tracks_other(self, fname, dname, fid, original_rgb_shape):
        dataset_dict = {}
        if self.load_tracks == 'tapir-sall':
            track_path = (self.basepath / 'Tracks' / 'tapir_ouputs_davis_pixeldense_31frames' / dname / fname).with_suffix('.npz')

            context_len = self.cfg.CTX_LEN
            
            r = np.load(str(track_path))
            tracks = torch.from_numpy(r['tracks']).float()  # N, f, 2
            vis = torch.from_numpy(r['visibles']).bool()  # N, f
            current_frame_id = r['eff_fid'].reshape(-1)[0]

            N, nframes, _ = tracks.shape
            if context_len > 0:
                start_index = current_frame_id - context_len
                end_index = current_frame_id + context_len + 1
                tracks = tracks[:, start_index:end_index]
                vis = vis[:, start_index:end_index]
                nframes = tracks.shape[1]
            
            f = nframes
            tracks = tracks.permute(1, 0, 2).reshape(nframes, N, 1, 2) # f, N, 2
            vis = vis.permute(1, 0).unsqueeze(2).bool()  # f, N, 1

            frame_H, frame_W, _ = original_rgb_shape

            loc = tracks[context_len] 
            loc[..., 0] /= frame_W-1
            loc[..., 1] /= frame_H-1
            loc = loc * 2 - 1
            loc.clamp_(-20, 20)
            loc = loc.view(1, -1, 2)  # 1, (hw), 2

            mean_vis = vis.float().mean(0).view(-1)
            all_vis = (mean_vis > self.cfg.TRACK_VIS_THRESH).float()
            # all_vis = torch.ones_like(all_vis)
            
            weights = torch.rand_like(all_vis) * all_vis 
            N = min(self.cfg.NTRACKS, N)
            _, idx = weights.topk(N, dim=0, largest=True, sorted=True)

            tracks = tracks[:, idx]
            tracks = (tracks.reshape(f, N, 2))
            if self.cfg.NORM == 'norm':
                tracks[..., 0] /= frame_W
                tracks[..., 1] /= frame_H  
                tracks.clamp_(-20, 20)
            else:
                tracks[..., 0].clamp_(-20 * frame_W, 20 * frame_W)
                tracks[..., 1].clamp_(-20 * frame_H, 20 * frame_H)
                
            # tracks = tracks.permute(2,0,3,1).reshape(1, f*2, N) # 1, f*2, N

            vis = vis[:, idx].contiguous()  # f N 1


            loc = loc[:, idx]  # 1, N, 2

            dataset_dict["track"] = tracks
            dataset_dict["vis"] = vis
            dataset_dict["loc"] = loc

        elif self.load_tracks == 'pips-sall':
            track_path = self.basepath / 'Tracks' / 'pips2_maxframes32'
            track_path = track_path / f"{dname}-{fname}.pt"

            context_len = self.cfg.CTX_LEN
            # cache_dir = f"cache_context_len_{context_len}"
            # cache_path = track_path.parent / cache_dir / track_path.name
            frame_H, frame_W, _ = original_rgb_shape

            # if self.cfg.USE_CACHE and cache_path.exists():
            #     r = torch.load(cache_path)
            #     tracks = r['tracks'].float()
            #     vis = r['vis'].bool()
            #     current_frame_id = r['eff_fid']
            #     frame_H = r.get('height', frame_H)
            #     frame_W = r.get('width', frame_W)

            # else:
            r = torch.load(track_path)
            tracks = r['tracks'].float()
            # vis = r['vis'].bool()
            current_frame_id = r['fid']
            frame_H = r.get('height', frame_H)
            frame_W = r.get('width', frame_W)

            nframes = tracks.shape[0]
            f, h, w, c = tracks.shape
            
            if context_len > 0:
                # vis = vis.reshape(f, h, w, 1)
                tracks = tracks.reshape(f, h, w, 2)

                start_index = current_frame_id - context_len
                end_index = current_frame_id + context_len + 1
                pad_before = 0
                if start_index < 0:
                    pad_before = -start_index
                    start_index = 0
                pad_after = 0
                if end_index > nframes:
                    pad_after = end_index - nframes
                    end_index = nframes
                
                tracks = tracks[start_index:end_index].float()
                # vis = vis[start_index:end_index].float()

                if pad_before or pad_after:
                    tracks = tracks.contiguous().numpy()
                    # vis = vis.contiguous().numpy()
                    tracks = np.pad(tracks, [(pad_before, pad_after), (0,0), (0,0), (0,0)], mode='reflect')
                    # vis = np.pad(vis, [(pad_before, pad_after), (0,0), (0,0), (0,0)], mode='reflect')
                    tracks = torch.from_numpy(tracks)
                    # vis = torch.from_numpy(vis)
                
                f = context_len*2 + 1
                current_frame_id = context_len
            
            f, h, w, c = tracks.shape
            tracks = tracks.view(f, h*w, 1, 2).float()
            f, ntracks = tracks.shape[:2]
            vis = torch.ones(f, ntracks, 1, dtype=bool)

            loc = tracks[current_frame_id] 
            loc[..., 0] /= frame_W-1
            loc[..., 1] /= frame_H-1
            loc = loc * 2 - 1
            loc.clamp_(-20, 20)
            loc = loc.view(1, ntracks, 2)  # 1, (hw), 2

            mean_vis = vis.float().mean(0).view(-1)
            all_vis = (mean_vis > self.cfg.TRACK_VIS_THRESH).float()
            
            
            # weights = torch.rand_like(all_vis) * all_vis 
            # _, idx = weights.topk(min(self.cfg.NTRACKS, ntracks), dim=0, largest=True, sorted=True)  # might also take points that are not visible in the end
            # if self.cfg.NTRACKS > ntracks:
            #     # require more tracks than exists
            #     # so take all
            #     fill_idx = torch.multinomial(all_vis.float(), self.cfg.NTRACKS - ntracks, replacement=True)
            #     idx = torch.cat([fill_idx, idx], dim=0) # put potentially invisible points at the end. They might get excluded when clipping the model. 


            n_valid_tracks = all_vis.sum().long().item()
            if n_valid_tracks < self.cfg.NTRACKS:
                idx = torch.arange(ntracks)[all_vis.bool()]
                fill_idx = torch.multinomial(all_vis.float(), self.cfg.NTRACKS - n_valid_tracks, replacement=True)
                idx = torch.cat([fill_idx, idx], dim=0)
            if n_valid_tracks > self.cfg.NTRACKS:
                idx = torch.multinomial(all_vis.float(), self.cfg.NTRACKS, replacement=False)
            

            N = len(idx)
            tracks = tracks[:, idx]
            vis = vis[:, idx]  # f N 1
            loc = loc[:, idx]  # 1, N, 2

            tracks = (tracks.reshape(f, N, 2))
            if self.cfg.NORM == 'norm':
                tracks[..., 0] /= frame_W -1
                tracks[..., 1] /= frame_H -1
                tracks.clamp_(-20, 20)
            else:
                tracks[..., 0].clamp_(-20 * frame_W, 20 * frame_W)
                tracks[..., 1].clamp_(-20 * frame_H, 20 * frame_H)
                
            # tracks = tracks.permute(2,0,3,1).reshape(1, f*2, N) # 1, f*2, N
            vis = vis.contiguous()  # f N 1
            loc = loc.contiguous()  # 1, N, 2

            vis_mass = (vis.float().mean(0).view(-1) > self.cfg.TRACK_VIS_THRESH).float().sum()

            dataset_dict["track"] = tracks
            dataset_dict["vis"] = vis
            dataset_dict["loc"] = loc

    def build_frame_item(self, fid, seq, rgb_path_offset, fwd_flow_path_offset, should_flip=False):
        data = {
            'frame_id': fid,
            'seq': seq
        }
        rgb, original_rgb = self.load_rgb(rgb_path_offset)
        data['rgb'] = rgb
        data['flow'] = self.load_dotflo(fwd_flow_path_offset)
        ano_path = self.metadata[seq][fid].get('Annotations', None)
        if ano_path is not None:
            ano = self.load_ano(ano_path)
            data['sem_seg'] = F.interpolate(ano.view(1,1,*ano.shape[-2:]).float(), mode='nearest', size=rgb.shape[-2:])[0, 0].long()
        else:
            data['sem_seg'] = torch.zeros((rgb.shape[-2:]), dtype=torch.long)
        # if data['flow'].shape[-2:] != data['rgb'].shape[-2:]:
        #     data['flow_big'] = data['flow']
        #     data['flow'] = self.load_dotflo(fwd_flow_path_offset, res=data['rgb'].shape[-2:])
            
        if should_flip:
            data['rgb'] = data['rgb'].flip(-1)
            data['flow'] = data['flow'].flip(-1)
            data['sem_seg'] = data['sem_seg'].flip(-1)

        fname = self.metadata[seq][fid]['fname']
        # print(seq, fid, fname)
        if self.load_tracks.startswith('co2') or self.load_tracks.startswith('boost'):
            track_dict = self.tracks[seq].get_by_fname(fname, should_flip)
            data.update(track_dict)
        elif self.load_tracks.startswith('tapir-sall') or self.load_tracks.startswith('pips-sall'):
            track_dict = self.load_tracks_other(fname, seq, fid, original_rgb.shape[-2:])
            data.update(track_dict)
        else:
            logger.info_once(f"Tracks not loaded: {self.load_tracks=}")
        
        return [data]

    def build_item(self, item):
        fid1, fid2, seq, gap, rgb1_path_offset, rgb2_path_offset, flow_fwd_path_off, flow_bwd_path_off = item

        should_flip = False
        if self.cfg.RANDOM_FLIP and (torch.rand(1) < 0.5).all():
            should_flip = True

        first_data1 = self.build_frame_item(fid1, seq, rgb1_path_offset, flow_fwd_path_off, should_flip)
        first_data2 = self.build_frame_item(fid2, seq, rgb2_path_offset, flow_bwd_path_off, should_flip)

        offset = 5

        offset_fid2 = fid2 + offset
        offset_fid1 = fid1 + offset
        if not offset_fid2 in self.metadata[seq]:
            offset_fid1 = fid1 - offset
            offset_fid2 = fid2 - offset
            assert offset_fid1 in self.metadata[seq]
            assert offset_fid2 in self.metadata[seq]
            second_data1 = first_data1
            second_data2 = first_data2
            rgb1_path_offset = self.metadata[seq][offset_fid1]['rgb']
            flow_fwd_path_off = self.metadata[seq][offset_fid1]['Flows_gap'][gap]

            rgb2_path_offset = self.metadata[seq][offset_fid2]['rgb']
            flow_bwd_path_off = self.metadata[seq][offset_fid2]['Flows_gap'][-gap]

            first_data1 = self.build_frame_item(offset_fid1, seq, rgb1_path_offset, flow_fwd_path_off, should_flip)
            first_data2 = self.build_frame_item(offset_fid2, seq, rgb2_path_offset, flow_bwd_path_off, should_flip)
        else:
            rgb1_path_offset = self.metadata[seq][offset_fid1]['rgb']
            flow_fwd_path_off = self.metadata[seq][offset_fid1]['Flows_gap'][gap]

            rgb2_path_offset = self.metadata[seq][offset_fid2]['rgb']
            flow_bwd_path_off = self.metadata[seq][offset_fid2]['Flows_gap'][-gap]

            second_data1 = self.build_frame_item(offset_fid1, seq, rgb1_path_offset, flow_fwd_path_off, should_flip)
            second_data2 = self.build_frame_item(offset_fid2, seq, rgb2_path_offset, flow_bwd_path_off, should_flip)

        # F, F+g, F+g+offset, F+g+offset+g
        return first_data1 + first_data2 + second_data1 + second_data2


def build_pair(cfg, basepath, img_res='480p', flow_res='1080p', ano_res='480p', flow_gaps=(1, 2), ano_paths=('Annotations',), seqs=None):
    basepath = basepath.lstrip('/')
    dataset_path = Path(cfg.GWM.DATA_ROOT) / basepath
    logger.info(f"Building pair dataset {dataset_path} {img_res=} {flow_res=} {ano_res=} {flow_gaps=} {ano_paths=}")
    meta = build_metadata(dataset_path, 
                          img_res, 
                          flow_dirs=("Flows_gap",), 
                          flow_gaps=flow_gaps, 
                          flow_res=flow_res,
                          ano_paths=ano_paths, 
                          ano_res=ano_res)
    if seqs is not None:
        logger.info(f'Filtering sequences: {seqs}')
        new_meta = {}
        for seq in seqs:
            new_meta[seq] = meta[seq]
        meta = new_meta
    
    logger.info(f"Built metadata for {len(meta)} sequences")
    logger.info(f"Example: {meta[next(iter(meta))][0]}")

    if cfg.FLAGS.INF_TPS:
        img_transforms = T.Compose([
            T.Resize(cfg.GWM.RESOLUTION, interpolation=T.InterpolationMode.LANCZOS),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
        ])
    else:
        img_transforms = T.Compose([
            T.Resize(cfg.GWM.RESOLUTION, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
    train_dataset = FlowTrackPairDataset(cfg, 
                                         meta, 
                                         dataset_path, 
                                         img_transforms, 
                                         flow_resolution=cfg.GWM.FLOW_RES, 
                                         flow_clip=cfg.GWM.FLOW_CLIP, 
                                         load_tracks=cfg.TRACKS,
                                         flow_gaps=flow_gaps,
                                         ano_types=ano_paths)
    return train_dataset