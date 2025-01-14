import math
import os
import random
from pathlib import Path

import detectron2.data.transforms as DT
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.data import detection_utils as d2_utils
from PIL import Image
from torch.utils.data import Dataset

from utils.data import read_flow
from utils.log import getLogger

logger = getLogger(__name__)


class TrackDataNoSH:
    def __init__(self, track_path, seq, n, step=4, threshold=0.1, sampling="random", frame_name_path=None):
        self.track_path = track_path
        self.seq = seq
        self.n = n
        self.threshold = threshold
        self.sampling = sampling
        self.frame_names = sorted([f.stem for f in (frame_name_path / seq).glob("*.jpg") if f.is_file()])
        if frame_name_path is not None:
            tracks = [self.track_path / f"{seq}-{f}.pt" for f in self.frame_names]
        else:
            tracks = sorted([f for f in (self.track_path).glob(f"{self.seq}-00???.pt") if f.is_file()])

        self.tracks = tracks
        self.f = len(self.tracks)
        self.T = torch.empty(self.f, 0, 2)
        archive = torch.load(self.tracks[0], map_location="cpu")
        self.h = archive.get("height", 480)
        self.w = archive.get("width", 854)

    def get_by_fname(self, fname):
        idx = self.frame_names.index(fname)
        return self[idx]

    def __len__(self):
        return self.f

    def __getitem__(self, idx):
        logger.info_once("Using tracks from 1 frame NOSHMEM")
        tpath = self.tracks[idx]
        archive = torch.load(tpath, map_location="cpu")
        v = archive["vis"].bool()
        t = archive["tracks"].half()
        if v.shape[0] != self.f:
            vt = torch.zeros(self.f, *v.shape[1:], dtype=v.dtype)
            tt = torch.zeros(self.f, *t.shape[1:], dtype=t.dtype)
            fid = archive["fid"]

            bv = v[:fid]
            av = v[fid:]
            s = idx - bv.shape[0]
            e = s + bv.shape[0] + av.shape[0]
            vt[s:e] = v
            tt[s:e] = t
            assert torch.all(vt[idx] == v[fid])
            assert torch.all(tt[idx] == t[fid])
            # print(idx, fid, tpath.name, v.shape, t.shape, vt.shape, tt.shape)
            v = vt
            t = tt

        v = v.view(self.f, -1)
        m = v.float().mean(0) > self.threshold
        V = v[:, m]
        T = t.view(self.f, -1, 2)[:, m, :]

        ntracks = T.shape[1]
        mean_vis = V.float().mean(0).view(-1)
        all_vis = (mean_vis > self.threshold).float()
        n_valid_tracks = all_vis.sum().long().item()
        if n_valid_tracks < self.n:
            index = torch.arange(ntracks)[all_vis.bool()]
            fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
            index = torch.cat([fill_index, index], dim=0)
        if n_valid_tracks >= self.n:
            index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class MutliframeTrackDataNoSH(TrackDataNoSH):
    def __getitem__(self, idx):
        logger.info_once("Using tracks from 3 frames NOSHMEM")
        if idx == 0:
            ids = [idx, idx + 1, idx + 2]
        elif idx == self.f - 1:
            ids = [idx - 2, idx - 1, idx]
        else:
            ids = [idx - 1, idx, idx + 1]

        tracks = []
        vis = []
        for i in ids:
            tpath = self.tracks[i]
            archive = torch.load(tpath, map_location="cpu")
            v = archive["vis"].bool()
            t = archive["tracks"].half()
            if v.shape[0] != self.f:
                vt = torch.zeros(self.f, *v.shape[1:], dtype=v.dtype)
                tt = torch.zeros(self.f, *t.shape[1:], dtype=t.dtype)
                fid = archive["fid"]

                bv = v[:fid]
                av = v[fid:]
                s = i - bv.shape[0]
                e = s + bv.shape[0] + av.shape[0]
                vt[s:e] = v
                tt[s:e] = t
                assert torch.all(vt[i] == v[fid])
                assert torch.all(tt[i] == t[fid])
                # print(idx, fid, tpath.name, v.shape, t.shape, vt.shape, tt.shape)
                v = vt
                t = tt
            vis.append(v.view(self.f, -1))
            t = t.view(self.f, -1, 2)
            tracks.append(t)
        v = torch.cat(vis, dim=1)
        t = torch.cat(tracks, dim=1)

        vis = v[idx]
        v = v[:, vis]
        t = t[:, vis]

        m = v.float().mean(0) > self.threshold
        V = v[:, m]
        T = t.view(self.f, -1, 2)[:, m, :]

        ntracks = T.shape[1]
        mean_vis = V.float().mean(0).view(-1)
        all_vis = (mean_vis > self.threshold).float()
        n_valid_tracks = all_vis.sum().long().item()
        if n_valid_tracks < self.n:
            index = torch.arange(ntracks)[all_vis.bool()]
            fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
            index = torch.cat([fill_index, index], dim=0)
        if n_valid_tracks >= self.n:
            index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class Mutliframe5TrackDataNoSH(TrackDataNoSH):
    def __getitem__(self, idx):
        logger.info_once("Using tracks from 5 frames NOSHMEM")
        if idx == 0:
            ids = [idx, idx + 1, idx + 2, idx + 3, idx + 4]
        if idx == 1:
            ids = [idx - 1, idx, idx + 1, idx + 2, idx + 3]
        elif idx == self.f - 1:
            ids = [idx - 4, idx - 3, idx - 2, idx - 1, idx]
        elif idx == self.f - 2:
            ids = [idx - 3, idx - 2, idx - 1, idx, idx + 1]

        else:

            ids = [idx - 2, idx - 1, idx, idx + 1, idx + 2]

        tracks = []
        vis = []
        for i in ids:
            tpath = self.tracks[i]
            archive = torch.load(tpath, map_location="cpu")
            v = archive["vis"].bool()
            t = archive["tracks"].half()
            if v.shape[0] != self.f:
                vt = torch.zeros(self.f, *v.shape[1:], dtype=v.dtype)
                tt = torch.zeros(self.f, *t.shape[1:], dtype=t.dtype)
                fid = archive["fid"]

                bv = v[:fid]
                av = v[fid:]
                s = i - bv.shape[0]
                e = s + bv.shape[0] + av.shape[0]
                vt[s:e] = v
                tt[s:e] = t
                assert torch.all(vt[i] == v[fid])
                assert torch.all(tt[i] == t[fid])
                # print(idx, fid, tpath.name, v.shape, t.shape, vt.shape, tt.shape)
                v = vt
                t = tt

            vis.append(v.view(self.f, -1))
            t = t.view(self.f, -1, 2)
            tracks.append(t)
        v = torch.cat(vis, dim=1)
        t = torch.cat(tracks, dim=1)

        vis = v[idx]
        v = v[:, vis]
        t = t[:, vis]

        m = v.float().mean(0) > self.threshold
        V = v[:, m]
        T = t.view(self.f, -1, 2)[:, m, :]

        ntracks = T.shape[1]
        mean_vis = V.float().mean(0).view(-1)
        all_vis = (mean_vis > self.threshold).float()
        n_valid_tracks = all_vis.sum().long().item()
        if n_valid_tracks < self.n:
            index = torch.arange(ntracks)[all_vis.bool()]
            fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
            index = torch.cat([fill_index, index], dim=0)
        if n_valid_tracks >= self.n:
            index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class TrackData:
    def __init__(self, track_path, seq, n, step=4, threshold=0.1, sampling="random", frame_name_path=None):
        self.track_path = track_path
        self.seq = seq
        self.n = n
        self.threshold = threshold
        self.sampling = sampling
        self.frame_names = sorted([f.stem for f in (frame_name_path / seq).glob("*.jpg") if f.is_file()])
        if frame_name_path is not None:
            tracks = [self.track_path / f"{seq}-{f}.pt" for f in self.frame_names]
        else:
            tracks = sorted([f for f in (self.track_path).glob(f"{self.seq}-00???.pt") if f.is_file()])

        f = len(tracks)
        self.f = f
        # archive = torch.load(self.track_path / f'{self.seq}_t0.5.pt')
        V = []
        T = []
        I = []
        for fid, tpath in enumerate(tracks):
            archive = torch.load(tpath, map_location="cpu")
            v = archive["vis"].bool()
            t = archive["tracks"].half()
            if v.shape[0] != self.f:
                vt = torch.zeros(self.f, *v.shape[1:], dtype=v.dtype)
                tt = torch.zeros(self.f, *t.shape[1:], dtype=t.dtype)
                fid_ = archive["fid"]

                bv = v[:fid_]
                av = v[fid_:]
                s = fid - bv.shape[0]
                e = s + bv.shape[0] + av.shape[0]
                vt[s:e] = v
                tt[s:e] = t
                assert torch.all(vt[fid] == v[fid_])
                assert torch.all(tt[fid] == t[fid_])
                # print(idx, fid, tpath.name, v.shape, t.shape, vt.shape, tt.shape)
                v = vt
                t = tt

            # print(fid, tpath.name, v.shape, t.shape)
            v = v.view(f, -1)
            m = v.float().mean(0) > threshold
            v = v[:, m]
            t = t.view(f, -1, 2)[:, m, :]
            V.append(v)
            T.append(t)
            I.append(torch.full((t.shape[1],), fid, dtype=torch.long))
        self.T = torch.cat(T, dim=1)
        self.V = torch.cat(V, dim=1)
        self.I = torch.cat(I, dim=0)

        self.h = archive.get("height", 480)
        self.w = archive.get("width", 854)
        self.f = f

        self.T.share_memory_()
        self.V.share_memory_()
        self.I.share_memory_()

        self.step = step

        if self.sampling == "stratified":
            max_cells_x = int(np.ceil(self.w / self.step) / self.step)
            tx = (self.T[:, :, 0].float() / self.step).floor().int()
            ty = (self.T[:, :, 1].float() * max_cells_x).floor().int()
            self.C = (tx + ty).int()
            self.C.share_memory_()

    def get_by_fname(self, fname):
        idx = self.frame_names.index(fname)
        return self[idx]

    def __len__(self):
        return self.f

    def __getitem__(self, idx):
        logger.info_once("Using tracks from 1 frame")
        if self.sampling == "stratified":
            sel = self.I == idx
            C = self.C[idx, sel]  # n
            T = self.T[:, sel]
            V = self.V[:, sel]

            shuffle = torch.randperm(len(C))
            _, unique_indices = np.unique(C[shuffle], return_index=True)
            index = shuffle[unique_indices[: self.n]]
            while len(index) < self.n:  # if there is frame with fewer points...
                index = torch.cat([index, shuffle[: self.n - len(index)]])

        if self.sampling == "random":
            sel = self.I == idx
            T = self.T[:, sel]
            V = self.V[:, sel]
            ntracks = T.shape[1]
            mean_vis = V.float().mean(0).view(-1)
            all_vis = (mean_vis > self.threshold).float()
            n_valid_tracks = all_vis.sum().long().item()
            if n_valid_tracks < self.n:
                index = torch.arange(ntracks)[all_vis.bool()]
                fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
                index = torch.cat([fill_index, index], dim=0)
            if n_valid_tracks >= self.n:
                index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class MultiTrackData(TrackData):
    def __getitem__(self, idx):
        logger.info_once("Using tracks from 3 frames")
        if idx == 0:
            ids = [idx, idx + 1, idx + 2]
        elif idx == self.f - 1:
            ids = [idx - 2, idx - 1, idx]
        else:
            ids = [idx - 1, idx, idx + 1]

        sel = self.I == ids[0]
        for i in ids[1:]:
            sel = torch.logical_or(self.I == i, sel)

        if self.sampling == "stratified":

            C = self.C[idx, sel]  # n
            T = self.T[:, sel]
            V = self.V[:, sel]

            # only visible in the current frame
            vis = V[idx]
            C = C[vis]
            T = T[:, vis]
            V = V[:, vis]

            shuffle = torch.randperm(len(C))
            _, unique_indices = np.unique(C[shuffle], return_index=True)
            index = shuffle[unique_indices[: self.n]]
            while len(index) < self.n:  # if there is frame with fewer points...
                index = torch.cat([index, shuffle[: self.n - len(index)]])

        if self.sampling == "random":

            T = self.T[:, sel]
            V = self.V[:, sel]

            vis = V[idx]
            V = V[:, vis]
            T = T[:, vis]

            ntracks = T.shape[1]
            mean_vis = V.float().mean(0).view(-1)
            all_vis = (mean_vis > self.threshold).float()
            n_valid_tracks = all_vis.sum().long().item()
            if n_valid_tracks < self.n:
                index = torch.arange(ntracks)[all_vis.bool()]
                fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
                index = torch.cat([fill_index, index], dim=0)
            if n_valid_tracks >= self.n:
                index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class Multi5TrackData(TrackData):
    def __getitem__(self, idx):
        logger.info_once("Using tracks from 5")
        if idx == 0:
            ids = [idx, idx + 1, idx + 2, idx + 3, idx + 4]
        if idx == 1:
            ids = [idx - 1, idx, idx + 1, idx + 2, idx + 3]
        elif idx == self.f - 1:
            ids = [idx - 4, idx - 3, idx - 2, idx - 1, idx]
        elif idx == self.f - 2:
            ids = [idx - 3, idx - 2, idx - 1, idx, idx + 1]

        else:

            ids = [idx - 2, idx - 1, idx, idx + 1, idx + 2]

        sel = self.I == ids[0]
        for i in ids[1:]:
            sel = torch.logical_or(self.I == i, sel)

        if self.sampling == "stratified":

            C = self.C[idx, sel]  # n
            T = self.T[:, sel]
            V = self.V[:, sel]

            # only visible in the current frame
            vis = V[idx]
            C = C[vis]
            T = T[:, vis]
            V = V[:, vis]

            shuffle = torch.randperm(len(C))
            _, unique_indices = np.unique(C[shuffle], return_index=True)
            index = shuffle[unique_indices[: self.n]]
            while len(index) < self.n:  # if there is frame with fewer points...
                index = torch.cat([index, shuffle[: self.n - len(index)]])

        if self.sampling == "random":

            T = self.T[:, sel]
            V = self.V[:, sel]

            vis = V[idx]
            V = V[:, vis]
            T = T[:, vis]

            ntracks = T.shape[1]
            mean_vis = V.float().mean(0).view(-1)
            all_vis = (mean_vis > self.threshold).float()
            n_valid_tracks = all_vis.sum().long().item()
            if n_valid_tracks < self.n:
                index = torch.arange(ntracks)[all_vis.bool()]
                fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
                index = torch.cat([fill_index, index], dim=0)
            if n_valid_tracks >= self.n:
                index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


class BWTrackData:
    def __init__(self, track_path, seq, n, step=4, threshold=0.1, sampling="random", reps=5, frame_name_path=None):
        self.track_path = track_path
        self.seq = seq
        self.n = n
        self.threshold = threshold
        self.sampling = sampling
        self.frame_names = sorted([f.stem for f in (frame_name_path / seq).glob("*.jpg") if f.is_file()])

        V = []
        T = []
        for rep_id in range(reps):
            archive = torch.load(self.track_path / f"{self.seq}_{rep_id}.pt")[0]
            V.append(archive["vis"].bool())
            T.append(archive["tracks"].half())

        T = torch.cat(T, dim=1)
        V = torch.cat(V, dim=1)

        suff_vis = V.float().mean(0) > threshold
        self.T = T[:, suff_vis].half()
        self.V = V[:, suff_vis].bool()

        self.h = archive.get("height", 480)
        self.w = archive.get("width", 854)
        self.f = T.shape[0]

        self.T.share_memory_()
        self.V.share_memory_()

        self.step = step

        logger.info(f"<{seq}> smallest frame {V.sum(-1).min()}")

        if self.sampling == "stratified":
            max_cells_x = int(np.ceil(self.w / self.step) / self.step)
            tx = (self.T[:, :, 0].float() / self.step).floor().int()
            ty = (self.T[:, :, 1].float() * max_cells_x).floor().int()
            self.C = (tx + ty).int()
            self.C.share_memory_()

    def get_by_fname(self, fname):
        idx = self.frame_names.index(fname)
        return self[idx]

    def __len__(self):
        return self.f

    def __getitem__(self, idx):
        if self.sampling == "stratified":
            sel = self.I == idx
            C = self.C[idx, sel]  # n
            T = self.T[:, sel]
            V = self.V[:, sel]

            shuffle = torch.randperm(len(C))
            _, unique_indices = np.unique(C[shuffle], return_index=True)
            index = shuffle[unique_indices[: self.n]]
            while len(index) < self.n:  # if there is frame with fewer points...
                index = torch.cat([index, shuffle[: self.n - len(index)]])

        if self.sampling == "random":
            sel = self.V[idx]

            T = self.T[:, sel]
            V = self.V[:, sel]

            ntracks = T.shape[1]
            mean_vis = V.float().mean(0).view(-1)
            all_vis = (mean_vis > self.threshold).float()
            n_valid_tracks = all_vis.sum().long().item()
            if n_valid_tracks < self.n:
                index = torch.arange(ntracks)[all_vis.bool()]
                fill_index = torch.multinomial(all_vis.float(), self.n - n_valid_tracks, replacement=True)
                index = torch.cat([fill_index, index], dim=0)
            if n_valid_tracks >= self.n:
                index = torch.multinomial(all_vis.float(), self.n, replacement=False)

        tracks = T[:, index].clone().float()
        vis = V[:, index].clone().float()

        return tracks, vis


def mov_avg_filt(t, n=3, dim=-1):
    if dim != -1:
        t = t.transpose(-1, dim)
    tshape = t.shape
    t = t.reshape(-1, tshape[-1])
    t = F.avg_pool1d(t, n, stride=1, padding=n // 2, count_include_pad=False)
    t = t.reshape(tshape)
    if dim != -1:
        t = t.transpose(-1, dim)
    return t


class TrackDataset:
    def __init__(self, data, ctx_len, padmode="contant", filt=0, scale=1.0, shift=0.0, ctx_mode="cont"):
        self.data = data
        self.ctx_len = ctx_len
        self.padmode = padmode
        self.filt = filt
        self.scale = scale
        self.shift = shift
        self.ctx_mode = ctx_mode

    def __len__(self):
        return len(self.data)

    def get_by_fname(self, fname, flip=False):
        idx = self.data.frame_names.index(fname)
        return self.__getitem__(idx, flip=flip)

    def __getitem__(self, idx, flip=False):
        t, v = self.data[idx]
        # t: (f, n, 2)
        # v: (f, n)

        n = t.shape[1]
        f = t.shape[0]

        loc = t[idx].clone()  # n, 2
        if flip:
            loc[..., 0] = self.data.w - 1 - loc[..., 0]

        loc[..., 0] /= self.data.w - 1
        loc[..., 1] /= self.data.h - 1
        loc = loc * 2 - 1  #
        loc.nan_to_num_(nan=0, posinf=3, neginf=-3)
        loc.clamp_(-3, 3)

        t = t.float()
        t.nan_to_num_(nan=0, posinf=1e6, neginf=-1e6)
        lim = 4
        lim = 20
        t[..., 0].clamp_(-lim * self.data.w, lim * self.data.w)
        t[..., 1].clamp_(-lim * self.data.h, lim * self.data.h)
        if flip:
            t[..., 0] = self.data.w - 1 - t[..., 0]

        # t[..., 0] /= self.data.w - 1.
        # t[..., 1] /= self.data.h - 1.
        # t *= self.scale
        # t += self.shift

        v = v.bool()

        if self.filt > 0:
            t = mov_avg_filt(t, self.filt, dim=0)

        if self.ctx_mode == "cont":
            # takes nearby frames
            if self.ctx_len > 0:
                start_index = idx - self.ctx_len
                end_index = idx + self.ctx_len + 1
                pad_before = 0
                if start_index < 0:
                    pad_before = -start_index
                    start_index = 0
                pad_after = 0
                if end_index > self.data.f:
                    pad_after = end_index - self.data.f
                    end_index = self.data.f

                if self.padmode is not None:

                    t = t[start_index:end_index].float()
                    v = v[start_index:end_index].float()

                    if pad_before or pad_after:
                        t = F.pad(t[None], (0, 0, 0, 0, pad_before, pad_after), mode=self.padmode)[0]
                        v = F.pad(v[None, None], (0, 0, pad_before, pad_after), mode=self.padmode)[0, 0]

                    t = t.float()
                    v = v.bool()

                else:

                    if pad_before:
                        end_index += pad_before
                    if pad_after:
                        start_index -= pad_after

                    start_index = max(0, start_index)
                    end_index = min(self.data.f, end_index)
                    t = t[start_index:end_index].float()  # f N 2
                    v = v[start_index:end_index].float()  # f N
                    if t.shape[0] < self.ctx_len * 2 + 1:
                        pad_before = (self.ctx_len * 2 + 1 - t.shape[0]) // 2
                        pad_after = self.ctx_len * 2 + 1 - t.shape[0] - pad_before

                        t = F.pad(t[None], (0, 0, 0, 0, pad_before, pad_after), mode="reflect")[0]
                        v = F.pad(v.float()[None, None], (0, 0, pad_before, pad_after), mode="reflect")[0, 0]

                    t = t.float()
                    v = v.float()

                f = t.shape[0]
        elif self.ctx_mode == "random":
            assert self.ctx_len > 0
            num_f = self.ctx_len * 2 + 1
            indexes = torch.randperm(self.data.f)[:num_f]
            if not (indexes == idx).any():
                indexes[-1] = idx
            indexes = indexes.sort()[0]
            t = t[indexes].float()
            v = v[indexes].bool()
            if len(t) < num_f:
                pad_before = (num_f - len(t)) // 2
                pad_after = num_f - len(t) - pad_before
                t = F.pad(t, (0, 0, 0, 0, pad_before, pad_after), mode=self.padmode)
                v = F.pad(v.float(), (0, 0, pad_before, pad_after), mode=self.padmode)

            t = t.float()
            v = v.float()
            f = t.shape[0]
        elif self.ctx_mode == "strided":
            num_f = self.ctx_len * 2 + 1
            if self.data.f > 2 * num_f:
                # Try to pack long range context in by striding
                stride = self.data.f // num_f
                before_idxs = torch.arange(0, idx, stride)
                after_idxs = torch.arange(idx, self.data.f, stride)
                indexes = torch.cat([before_idxs, after_idxs])
                if len(indexes) < num_f:
                    known_idx = set(indexes.tolist())
                    missing = [i for i in range(self.data.f) if i not in known_idx]
                    missing = torch.tensor(missing)
                    fill_idx = missing[torch.randperm(missing.shape[0])[: (num_f - len(indexes))]]
                    indexes = torch.cat([indexes, fill_idx])
                elif len(indexes) > num_f:
                    indexes = indexes[:num_f]
                assert len(indexes) == num_f
                indexes = indexes.sort()[0]
            elif self.data.f >= num_f:
                # Try to get as many nearby ideas as possible
                before_idxs = torch.arange(max(0, idx - self.ctx_len), idx)
                after_idxs = torch.arange(idx, min(self.data.f, idx + self.ctx_len + 1))
                indexes = torch.cat([before_idxs, after_idxs])
                if len(indexes) < num_f:
                    known_idx = set(indexes.tolist())
                    missing = [i for i in range(self.data.f) if i not in known_idx]
                    missing = torch.tensor(missing)
                    fill_idx = missing[torch.randperm(missing.shape[0])[: (num_f - len(indexes))]]
                    indexes = torch.cat([indexes, fill_idx])
                indexes = indexes.sort()[0]
                assert len(indexes) == num_f
            else:
                # There is not enought, so take all and reflect pad
                indexes = torch.arange(self.data.f)
                pad_before = (num_f - len(indexes)) // 2
                pad_after = num_f - len(indexes) - pad_before
                # indexes = F.pad(indexes.float().view(1,1,1,-1), (pad_before, pad_after), mode='reflect').view(-1).long()
                indexes = (
                    torch.from_numpy(np.pad(indexes.numpy(), (pad_before, pad_after), mode="reflect")).view(-1).long()
                )
                assert len(indexes) == num_f
            t = t[indexes].float()
            v = v[indexes].bool()
            f = t.shape[0]

        assert t.shape == (f, n, 2)
        t = t.reshape(f, n, 2)
        assert v.shape == (f, n)
        v = v.reshape(f, n, 1)
        assert loc.shape == (n, 2)
        loc = loc.reshape(1, n, 2)

        dataset_dict = {}
        dataset_dict["track"] = t
        dataset_dict["vis"] = v
        dataset_dict["loc"] = loc
        return dataset_dict


class FlowPairDetectron(Dataset):
    def __init__(
        self,
        data_dir,
        resolution,
        to_rgb=False,
        size_divisibility=None,
        enable_photo_aug=False,
        flow_clip=1.0,
        norm=True,
        read_big=False,
        force1080p=False,
        flow_res=None,
        load_tracks="cotracker-all",
        cfg=None,
    ):
        self.eval = eval
        self.to_rgb = to_rgb
        self.data_dir = data_dir
        self.flow_dir = {k: [e for e in v if e.shape[0] > 0] for k, v in data_dir[0].items()}
        self.flow_dir = {k: v for k, v in self.flow_dir.items() if len(v) > 0}
        self.resolution = resolution
        self.size_divisibility = size_divisibility
        logger.info("Size divisibility: %s", self.size_divisibility)
        self.ignore_label = -1
        self.transforms = DT.AugmentationList(
            [
                DT.Resize(self.resolution, interp=Image.BICUBIC),
            ]
        )
        self.photometric_aug = (
            T.Compose(
                [
                    T.RandomApply(
                        torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]),
                        p=0.8,
                    ),
                    T.RandomGrayscale(p=0.2),
                ]
            )
            if enable_photo_aug
            else None
        )
        self.flow_clip = flow_clip
        self.norm_flow = norm
        self.read_big = read_big
        self.force1080p_transforms = None
        if force1080p:
            self.force1080p_transforms = DT.AugmentationList(
                [
                    DT.Resize((1088, 1920), interp=Image.BICUBIC),
                ]
            )
        self.big_flow_resolution = flow_res
        self.load_tracks = load_tracks
        self.cfg = cfg
        self.load_flow_conf = "-flow" in cfg.GWM.LOSS

        if self.load_tracks.startswith("co2") or self.load_tracks.startswith("boost"):
            seqs = sorted(list(set(a.name for a in self.data_dir[1].iterdir() if a.is_dir())))
            if self.load_tracks.startswith("co2"):
                track_path = self.data_dir[3] / "Tracks" / "cotrackerv2_rel_stride4_aux2"
                if "BW" in self.load_tracks:
                    track_path = self.data_dir[3] / "Tracks" / "cotrackerBW_12288"
            elif self.load_tracks.startswith("boost"):
                track_path = self.data_dir[3] / "Tracks" / "boosttapir_stride4"
            self.tracks = {}
            for seq in seqs:
                if "noshmem" in self.load_tracks:
                    if "multi5" in self.load_tracks:
                        data = Mutliframe5TrackDataNoSH(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                    elif "multi" in self.load_tracks:
                        data = MutliframeTrackDataNoSH(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                    else:
                        data = TrackDataNoSH(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                elif "BWcomb" in self.load_tracks:
                    data = BWTrackData(
                        track_path,
                        seq,
                        n=self.cfg.NTRACKS,
                        step=4,
                        threshold=self.cfg.TRACK_VIS_THRESH,
                        sampling="random",
                        reps=5,
                        frame_name_path=self.data_dir[1],
                    )
                elif "BW" in self.load_tracks:
                    data = BWTrackData(
                        track_path,
                        seq,
                        n=self.cfg.NTRACKS,
                        step=4,
                        threshold=self.cfg.TRACK_VIS_THRESH,
                        sampling="random",
                        reps=1,
                        frame_name_path=self.data_dir[1],
                    )
                else:
                    if "multi5" in self.load_tracks:
                        data = Multi5TrackData(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                    elif "multi" in self.load_tracks:
                        data = MultiTrackData(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                    else:
                        data = TrackData(
                            track_path,
                            seq,
                            n=self.cfg.NTRACKS,
                            step=4,
                            threshold=self.cfg.TRACK_VIS_THRESH,
                            sampling="random",
                            frame_name_path=self.data_dir[1],
                        )
                dataset = TrackDataset(
                    data,
                    ctx_len=self.cfg.CTX_LEN,
                    filt=self.cfg.FILTERN,
                    padmode="constant" if "zeropad" in self.load_tracks else None,
                    scale=1.,
                    shift=0,
                    ctx_mode=self.cfg.CTX_MODE,
                )
                self.tracks[seq] = dataset
                logger.info(f"Loaded and indexed <{seq}> tracks : {data.T.shape} {data.f}")

    def __len__(self):
        return (
            sum([cat.shape[0] for cat in next(iter(self.flow_dir.values()))]) if len(self.flow_dir.values()) > 0 else 0
        )

    def __getitem__(self, idx):

        dataset_dicts = []

        random_gap = random.choice(list(self.flow_dir.keys()))
        flowgaps = self.flow_dir[random_gap]
        vid = random.choice(flowgaps)
        flos = random.choice(vid)
        dataset_dict = {}

        fname = Path(flos[0]).stem  # frame number as a string
        dname = Path(flos[0]).parent.name  # sequence name
        suffix = ".png" if "CLEVR" in fname else ".jpg"
        rgb_dir = (self.data_dir[1] / dname / fname).with_suffix(suffix)
        gt_dir = (self.data_dir[2] / dname / fname).with_suffix(".png")

        flo_res = self.big_flow_resolution or self.resolution
        flo0 = einops.rearrange(read_flow(str(flos[0]), flo_res, self.to_rgb), "c h w -> h w c")
        if self.cfg.GWM.USE_MULT_FLOW:
            flo1 = einops.rearrange(read_flow(str(flos[1]), flo_res, self.to_rgb), "c h w -> h w c")
        rgb = d2_utils.read_image(rgb_dir).astype(np.float32)

        original_rgb_shape = rgb.shape

        original_rgb = torch.as_tensor(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)).clip(0.0, 255.0))).float()
        if self.read_big:
            rgb_big = d2_utils.read_image(str(rgb_dir).replace("480p", "1080p")).astype(np.float32)
            rgb_big = (torch.as_tensor(np.ascontiguousarray(rgb_big))[:, :, :3]).permute(2, 0, 1).clamp(0.0, 255.0)
            if self.force1080p_transforms is not None:
                rgb_big = F.interpolate(rgb_big[None], size=(1080, 1920), mode="bicubic").clamp(0.0, 255.0)[0]

        # print('not here', rgb.min(), rgb.max())
        input = DT.AugInput(rgb)

        # Apply the augmentation:
        preprocessing_transforms = self.transforms(input)  # type: DT.Transform
        rgb = input.image
        if self.photometric_aug:
            rgb_aug = Image.fromarray(rgb.astype(np.uint8))
            rgb_aug = self.photometric_aug(rgb_aug)
            rgb_aug = d2_utils.convert_PIL_to_numpy(rgb_aug, "RGB")
            rgb_aug = np.transpose(rgb_aug, (2, 0, 1)).astype(np.float32)
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = rgb.clip(0.0, 255.0)
        # print('here', rgb.min(), rgb.max())
        d2_utils.check_image_size(dataset_dict, flo0)
        if gt_dir.exists():
            sem_seg_gt = d2_utils.read_image(str(gt_dir))
            sem_seg_gt = preprocessing_transforms.apply_segmentation(sem_seg_gt)
            if sem_seg_gt.ndim == 3:
                sem_seg_gt = sem_seg_gt[:, :, 0]
            if sem_seg_gt.max() == 255:
                sem_seg_gt = (sem_seg_gt > 128).astype(int)
        else:
            sem_seg_gt = np.zeros((self.resolution[0], self.resolution[1]))

        gwm_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Pad image and segmentation label here!
        if self.to_rgb:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1))) / 2 + 0.5
            flo0 = flo0 * 255
            if self.cfg.GWM.USE_MULT_FLOW:
                flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1))) / 2 + 0.5
                flo1 = flo1 * 255

        else:
            flo0 = torch.as_tensor(np.ascontiguousarray(flo0.transpose(2, 0, 1)))
            if self.cfg.GWM.USE_MULT_FLOW:
                flo1 = torch.as_tensor(np.ascontiguousarray(flo1.transpose(2, 0, 1)))

            if self.norm_flow:
                flo0 = flo0 / (flo0**2).sum(0).max().sqrt()
                if self.cfg.GWM.USE_MULT_FLOW:
                    flo1 = flo1 / (flo1**2).sum(0).max().sqrt()

            flo0 = flo0.clip(-self.flow_clip, self.flow_clip)
            if self.cfg.GWM.USE_MULT_FLOW:
                flo1 = flo1.clip(-self.flow_clip, self.flow_clip)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb))
        if self.photometric_aug:
            rgb_aug = torch.as_tensor(np.ascontiguousarray(rgb_aug))

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        if gwm_seg_gt is not None:
            gwm_seg_gt = torch.as_tensor(gwm_seg_gt.astype("long"))

        image_shape = (rgb.shape[-2], rgb.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["seq"] = dname
        dataset_dict["flow"] = flo0
        if self.cfg.GWM.USE_MULT_FLOW:
            dataset_dict["flow_2"] = flo1

        dataset_dict["rgb"] = rgb
        if self.read_big:
            dataset_dict["RGB_BIG"] = rgb_big
        if self.photometric_aug:
            dataset_dict["rgb"] = rgb_aug
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if gwm_seg_gt is not None:
            dataset_dict["gwm_seg"] = gwm_seg_gt.long()

        should_flip = False
        if self.cfg.RANDOM_FLIP:
            should_flip = torch.rand(1).item() > 0.5
            if should_flip:
                pos_keys = [
                    "flow",
                    "flow_2",
                    "flow_big",
                    "flow_big_2",
                    "rgb",
                    "RGB_BIG",
                    "rgb_aug",
                    "original_rgb",
                    "sem_seg",
                    "gwm_seg",
                ]
                for k in pos_keys:
                    if k in dataset_dict:
                        t = dataset_dict[k]
                        logger.info(f"Flipping {k} {t.shape}")
                        t = t.flip(-1)
                        dataset_dict[k] = t

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        if self.load_tracks.startswith("co2") or self.load_tracks.startswith("boost"):
            # fid = int(fname.lstrip('0')) if fname.lstrip('0') != '' else 0
            seq = str(dname)
            track_dict = self.tracks[seq].get_by_fname(fname, should_flip)

            dataset_dict.update(track_dict)

        elif self.load_tracks == "tapir-sall":
            frame_id = int(fname.lstrip("0")) if fname.lstrip("0") != "" else 0
            track_path = (
                Path(str(self.data_dir[2]).replace("Annotations", "Tracks")).parent
                / "tapir_ouputs_davis_pixeldense_31frames"
                / dname
                / fname
            ).with_suffix(".npz")

            context_len = self.cfg.CTX_LEN

            r = np.load(str(track_path))
            tracks = torch.from_numpy(r["tracks"]).float()  # N, f, 2
            vis = torch.from_numpy(r["visibles"]).bool()  # N, f
            current_frame_id = r["eff_fid"].reshape(-1)[0]

            N, nframes, _ = tracks.shape
            if context_len > 0:
                start_index = current_frame_id - context_len
                end_index = current_frame_id + context_len + 1
                tracks = tracks[:, start_index:end_index]
                vis = vis[:, start_index:end_index]
                nframes = tracks.shape[1]

            f = nframes
            tracks = tracks.permute(1, 0, 2).reshape(nframes, N, 1, 2)  # f, N, 2
            vis = vis.permute(1, 0).unsqueeze(2).bool()  # f, N, 1

            frame_H, frame_W, _ = original_rgb_shape

            loc = tracks[context_len]
            loc[..., 0] /= frame_W - 1
            loc[..., 1] /= frame_H - 1
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
            tracks = tracks.reshape(f, N, 2)
            if self.cfg.NORM == "norm":
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

        elif self.load_tracks == "pips-sall":
            track_path = self.data_dir[3] / "Tracks" / "pips2_maxframes32"
            track_path = track_path / f"{dname}-{fname}.pt"

            context_len = self.cfg.CTX_LEN
            frame_H, frame_W, _ = original_rgb_shape

            r = torch.load(track_path)
            tracks = r["tracks"].float()
            current_frame_id = r["fid"]
            frame_H = r.get("height", frame_H)
            frame_W = r.get("width", frame_W)

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
                    tracks = np.pad(tracks, [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)], mode="reflect")
                    # vis = np.pad(vis, [(pad_before, pad_after), (0,0), (0,0), (0,0)], mode='reflect')
                    tracks = torch.from_numpy(tracks)
                    # vis = torch.from_numpy(vis)

                f = context_len * 2 + 1
                current_frame_id = context_len

            f, h, w, c = tracks.shape
            tracks = tracks.view(f, h * w, 1, 2).float()
            f, ntracks = tracks.shape[:2]
            vis = torch.ones(f, ntracks, 1, dtype=bool)

            loc = tracks[current_frame_id]
            loc[..., 0] /= frame_W - 1
            loc[..., 1] /= frame_H - 1
            loc = loc * 2 - 1
            loc.clamp_(-20, 20)
            loc = loc.view(1, ntracks, 2)  # 1, (hw), 2

            mean_vis = vis.float().mean(0).view(-1)
            all_vis = (mean_vis > self.cfg.TRACK_VIS_THRESH).float()

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

            tracks = tracks.reshape(f, N, 2)
            if self.cfg.NORM == "norm":
                tracks[..., 0] /= frame_W - 1
                tracks[..., 1] /= frame_H - 1
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

        elif self.load_tracks in ["cotracker2-sall", "cotracker2-sall-zeropad", "cotracker2-sall-zeropad-down"]:
            track_path = self.data_dir[3] / "Tracks" / "cotracker"
            if "DAVIS" in self.cfg.GWM.DATASET:
                frame_id = int(fname.lstrip("0")) if fname.lstrip("0") != "" else 0
                track_path = track_path / f"{dname}-{frame_id}.pt"
                del frame_id
            else:
                track_path = track_path / f"{dname}-{fname}.pt"

            context_len = self.cfg.CTX_LEN
            cache_dir = f"cache_context_len_{context_len}"
            cache_path = track_path.parent / cache_dir / track_path.name
            frame_H, frame_W, _ = original_rgb_shape

            r = torch.load(track_path)
            tracks = r["tracks"].float()
            vis = r["vis"].bool()
            if self.load_tracks == "cotracker2-sall-zeropad-down":
                tracks = tracks[:, 1::2, 1::2]
                vis = vis[:, 1::2, 1::2]

            current_frame_id = r["fid"]
            frame_H = r.get("height", frame_H)
            frame_W = r.get("width", frame_W)

            nframes = tracks.shape[0]
            f, h, w, c = tracks.shape

            loc = tracks[current_frame_id].clone()

            if self.cfg.NORM == "norm":
                tracks[..., 0] /= frame_W - 1
                tracks[..., 1] /= frame_H - 1
                tracks.clamp_(-20, 20)
            else:
                tracks[..., 0].clamp_(-20 * frame_W, 20 * frame_W)
                tracks[..., 1].clamp_(-20 * frame_H, 20 * frame_H)

            if self.cfg.FILTERN > 0:
                tracks = mov_avg_filt(tracks, self.cfg.FILTERN, dim=0)

            if context_len > 0:
                vis = vis.reshape(f, h, w, 1)
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
                vis = vis[start_index:end_index].float()

                mode = "constant" if "zeropad" in self.load_tracks else "reflect"
                if pad_before or pad_after:
                    tracks = tracks.contiguous().numpy()
                    vis = vis.contiguous().numpy()
                    tracks = np.pad(tracks, [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)], mode=mode)
                    vis = np.pad(vis, [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)], mode=mode)
                    tracks = torch.from_numpy(tracks)
                    vis = torch.from_numpy(vis)

                f = context_len * 2 + 1
                current_frame_id = context_len

            f, h, w, c = tracks.shape
            tracks = tracks.reshape(f, h * w, 1, 2).float()
            f, ntracks = tracks.shape[:2]
            vis = vis.reshape(f, ntracks, 1).bool()

            loc[..., 0] /= frame_W - 1
            loc[..., 1] /= frame_H - 1
            loc = loc * 2 - 1
            loc.clamp_(-20, 20)
            loc = loc.view(1, ntracks, 2)  # 1, (hw), 2

            mean_vis = vis.float().mean(0).view(-1)
            all_vis = (mean_vis > self.cfg.TRACK_VIS_THRESH).float()

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

            tracks = tracks.reshape(f, N, 2)

            # tracks = tracks.permute(2,0,3,1).reshape(1, f*2, N) # 1, f*2, N
            vis = vis.contiguous()  # f N 1
            loc = loc.contiguous()  # 1, N, 2

            vis_mass = (vis.float().mean(0).view(-1) > self.cfg.TRACK_VIS_THRESH).float().sum()

            dataset_dict["track"] = tracks
            dataset_dict["vis"] = vis
            dataset_dict["loc"] = loc

        elif self.load_tracks in [
            "cotracker2rel-sall-zeropad",
            "cotracker2relv2-sall-zeropad",
            "cotracker2relv2_s8-sall-zeropad",
            "cotracker2relv2comb-sall-zeropad",
        ]:
            folder = "cotrackerv2_rel_max50_stride4"
            if self.load_tracks == "cotracker2relv2-sall-zeropad":
                folder = "cotrackerv2_rel_stride4_aux2"
            if self.load_tracks == "cotracker2relv2_s8-sall-zeropad":
                folder = "cotrackerv2_rel_stride8_aux2"
            if self.load_tracks == "cotracker2relv2comb-sall-zeropad":
                folder = "cotrackerv2_rel_stride4_aux2_combined"

            track_path = self.data_dir[3] / "Tracks" / folder
            track_path = track_path / f"{dname}-{fname}.pt"

            context_len = self.cfg.CTX_LEN
            cache_dir = f"cache_context_len_{context_len}"
            cache_path = track_path.parent / cache_dir / track_path.name
            frame_H, frame_W, _ = original_rgb_shape

            r = torch.load(track_path)
            tracks = r["tracks"].float()
            vis = r["vis"].bool()
            current_frame_id = r["fid"]
            frame_H = r.get("height", frame_H)
            frame_W = r.get("width", frame_W)

            nframes = tracks.shape[0]
            f, h, w, c = tracks.shape

            loc = tracks[current_frame_id].clone()

            if self.cfg.NORM == "norm":
                tracks[..., 0] /= frame_W - 1
                tracks[..., 1] /= frame_H - 1
                tracks.clamp_(-20, 20)
            else:
                limW = 20 * (frame_W)
                limH = 20 * (frame_H)
                tracks[..., 0].clamp_(-limW, limW)
                tracks[..., 1].clamp_(-limH, limH)

            if self.cfg.FILTERN > 0:
                tracks = mov_avg_filt(tracks, self.cfg.FILTERN, dim=0)


            if context_len > 0:
                vis = vis.reshape(f, h, w, 1)
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
                vis = vis[start_index:end_index].float()

                if pad_before or pad_after:
                    tracks = tracks.contiguous().numpy()
                    vis = vis.contiguous().numpy()
                    tracks = np.pad(tracks, [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)], mode="constant")
                    vis = np.pad(vis, [(pad_before, pad_after), (0, 0), (0, 0), (0, 0)], mode="constant")
                    tracks = torch.from_numpy(tracks)
                    vis = torch.from_numpy(vis)

                f = context_len * 2 + 1
                current_frame_id = context_len
            else:
                new_tracks = torch.zeros((41, *tracks.shape[1:]))
                new_vis = torch.zeros((41, *vis.shape[1:]))
                new_tracks[:f] = tracks
                new_vis[:f] = vis
                tracks = new_tracks
                vis = new_vis
                f = 41

            f, h, w, c = tracks.shape
            tracks = tracks.view(f, h * w, 1, 2).float()
            f, ntracks = tracks.shape[:2]
            vis = vis.view(f, ntracks, 1).bool()

            loc[..., 0] /= frame_W - 1
            loc[..., 1] /= frame_H - 1
            loc = loc * 2 - 1
            loc.clamp_(-20, 20)
            loc = loc.view(1, ntracks, 2)  # 1, (hw), 2

            mean_vis = vis.float().mean(0).view(-1)
            all_vis = (mean_vis > self.cfg.TRACK_VIS_THRESH).float()
            if all_vis.sum() <= 0:
                logger.error(f"all_vis.sum() <= 0 for {track_path}")
                print(f"ERROR: all_vis.sum() <= 0 for {track_path}")
                all_vis = torch.ones_like(all_vis)

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
            tracks = tracks.reshape(f, N, 2)

            tracks = tracks.contiguous()  # f N 2
            vis = vis.contiguous()  # f N 1
            loc = loc.contiguous()  # 1, N, 2

            dataset_dict["track"] = tracks
            dataset_dict["vis"] = vis
            dataset_dict["loc"] = loc

        else:
            logger.debug_once(f"Not loading tracks: GWM.TRACKS={self.load_tracks}")

        dataset_dicts.append(dataset_dict)

        return dataset_dicts


def mov_avg_filt(t, n=3, dim=-1):
    if dim != -1:
        t = t.transpose(-1, dim)
    tshape = t.shape
    t = t.reshape(-1, tshape[-1])
    t = F.avg_pool1d(t, n, stride=1, padding=n // 2, count_include_pad=False)
    t = t.reshape(tshape)
    if dim != -1:
        t = t.transpose(-1, dim)
    return t
