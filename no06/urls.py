from django.urls import path
from . import views

urlpatterns = [
    path('',                   views.index,             name='index'),
    path('upload/',            views.upload_video,      name='upload_video'),
    path('selectVideos/',      views.select_videos,     name='select_videos'),
    path('generateData/',      views.generate_data,     name='generate_data'),
    path('make_distribution/', views.make_distribution, name='make_distribution'),
    path('showDistribution/',  views.show_distribution, name='show_distribution'),
    path('getImage/',          views.get_image,         name='get_image'),
    path('getVideo/',          views.get_video,         name='get_video'),
    path('getThumbList/',      views.get_thumb_list,    name='get_thumb_list'),

    # ★ 新規
    path('selectPose/',        views.select_pose,       name='select_pose'),
    path('poseEstimate/',      views.pose_estimate,     name='pose_estimate'),

    path('showDirectory/',     views.show_directory,    name='show_directory'),
]
