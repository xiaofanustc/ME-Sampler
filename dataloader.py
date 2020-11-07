import os
import gluoncv
from mxnet import nd

class SomethingSomethingV2_revise(gluoncv.data.SomethingSomethingV2):
    def __init__(self,
                 root=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/20bn-something-something-v2-frames'),
                 setting=os.path.expanduser('~/.mxnet/datasets/somethingsomethingv2/train_videofolder.txt'),
                 train=True,
                 test_mode=False,
                 name_pattern='%06d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 new_length=1,
                 new_step=1,
                 new_width=340,
                 new_height=256,
                 target_width=224,
                 target_height=224,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 transform=None):

        super(SomethingSomethingV2_revise, self).__init__(root=root,
                                                          setting=setting,
                                                          train=train,
                                                          test_mode=test_mode,
                                                          name_pattern=name_pattern,
                                                          video_ext=video_ext,
                                                          is_color=is_color,
                                                          modality=modality,
                                                          num_segments=num_segments,
                                                          new_length=new_length,
                                                          new_step=new_step,
                                                          new_width=new_width,
                                                          new_height=new_height,
                                                          target_width=target_width,
                                                          target_height=target_height,
                                                          temporal_jitter=temporal_jitter,
                                                          video_loader=video_loader,
                                                          use_decord=use_decord,
                                                          transform=transform)




    def __getitem__(self, index):

        directory, duration, target = self.clips[index]
        if self.video_loader:
            if self.use_decord:
                decord_vr = self.decord.VideoReader('{}.{}'.format(directory, self.video_ext), width=self.new_width, height=self.new_height)
                duration = len(decord_vr)
            else:
                mmcv_vr = self.mmcv.VideoReader('{}.{}'.format(directory, self.video_ext))
                duration = len(mmcv_vr)

        if self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_train_indices(duration)
        elif not self.train and not self.test_mode:
            segment_indices, skip_offsets = self._sample_val_indices(duration)
        else:
            segment_indices, skip_offsets = self._sample_test_indices(duration)

        # N frames of shape H x W x C, where N = num_oversample * num_segments * new_length
        if self.video_loader:
            if self.use_decord:
                clip_input = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            else:
                clip_input = self._video_TSN_mmcv_loader(directory, mmcv_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = self._image_TSN_cv2_loader(directory, duration, segment_indices, skip_offsets)

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        '''
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (self.new_length, 3, self.target_height, self.target_width))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

        if self.new_length == 1:
            clip_input = np.squeeze(clip_input, axis=2)    # this is for 2D input case
        '''
        return nd.array(clip_input), target, int(directory.split('/')[-1])


