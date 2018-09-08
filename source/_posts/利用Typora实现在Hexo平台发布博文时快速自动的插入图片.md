---
title: 利用Typora实现在Hexo平台发布博文时快速自动的插入图片
typora-copy-images-to: 利用Typora实现在Hexo平台发布博文时快速自动的插入图片
date: 2018-08-11 01:10:31
tags: 
- Hexo
- Typora
categories: 工具
---

## 开启post_asset_folder选项

编辑_config.yml文件，设置：
`post_asset_folder: true`

## 安装图片自动上传插件

终端中运行如下命令：
`npm install hexo-asset-image --save`

## 设置Typora选项

按如下图设置Typora，选择其中“允许根据YAML设置自动上传图片”的功能
![image-20180811011929179](利用Typora实现在Hexo平台发布博文时快速自动的插入图片/image-20180811011929179.png)

## 修改post、page、draft模版

参考下图，分别添加最后一行代码：![image-20180811012232175](利用Typora实现在Hexo平台发布博文时快速自动的插入图片/image-20180811012232175.png)

## 完毕！

接下来就是Happy的Writing时间，如果需要插入图片，直接在Typora中`Cmd+V`即可。:)