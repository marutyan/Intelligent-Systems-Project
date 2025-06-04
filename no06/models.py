from django.db import models


class Video(models.Model):
    """アップロード済み動画 1 本を表す"""
    name      = models.CharField(max_length=100)
    file      = models.CharField(max_length=200)          # BASE_DIR からの相対パス
    uploaded  = models.DateTimeField(auto_now_add=True)   # ← NULL 制約エラー対策で追加

    def __str__(self):
        return f'{self.id}:{self.name}'


class DistributionData(models.Model):
    """
    1 行 = 1 フレームの特徴量  
      video      – Video への FK  
      timestamp  – 動画内秒数  
      tsne_x/y/z – t-SNE 座標  
      cluster    – KMeans のクラスタ ID (0-4)  
      feature_vec– 12 次元 RGB ベクトル（JSON で保存）  
      thumb_path – サムネイル画像の相対パス
    """
    video        = models.ForeignKey(Video, on_delete=models.CASCADE)
    timestamp    = models.FloatField()
    tsne_x       = models.FloatField(default=0)
    tsne_y       = models.FloatField(default=0)
    tsne_z       = models.FloatField(default=0)

    # ★ ここが今回追加した 3 列
    cluster      = models.IntegerField(null=True, blank=True)
    feature_vec  = models.JSONField(null=True, blank=True)
    thumb_path   = models.CharField(max_length=200, null=True, blank=True)

    class Meta:
        unique_together = ('video', 'timestamp')

    def __str__(self):
        return f'{self.video_id}@{self.timestamp:.1f}s'
