//
//  FramesTableViewController.swift
//  lumos ios
//
//  Created by Johnathan Chen on 2/23/18.
//  Copyright © 2018 Shalin Shah. All rights reserved.
//

import UIKit
import AVFoundation
import AVKit
import CropViewController

class FramesTableViewController: UIViewController {
    var cancelButton: UIButton!

    var image = UIImage()
    var imageCollection = [UIImage]()
    
    // URL of video. URL is being passed from TenSecondViewController
    //private var videoURL: URL
    
    // The number of seconds the user captured
    //private var durationTime : Double
    
    //private var video : AVAsset
    //var assetImgGenerate : AVAssetImageGenerator

    init(imagesTaken: [UIImage]) {
        //self.videoURL = videoURL
        //self.video = AVAsset(url: videoURL)
        //self.durationTime = CMTimeGetSeconds(video.duration)
        //self.assetImgGenerate = AVAssetImageGenerator(asset: video)
        self.imageCollection = imagesTaken
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewWillAppear(_ animated: Bool) {
        self.navigationController?.isNavigationBarHidden = false
        
    }
    
    private var myTableView: UITableView!

//    func generateThumnail(url : URL, fromTime:Float64) -> UIImage {
////        assetImgGenerate.appliesPreferredTrackTransform = true
////        assetImgGenerate.requestedTimeToleranceAfter = kCMTimeZero;
////        assetImgGenerate.requestedTimeToleranceBefore = kCMTimeZero;
////        let error       : NSError? = nil
//
//        let time        : CMTime = CMTimeMakeWithSeconds(fromTime, 600)
//
//        let img         : CGImage = try! assetImgGenerate.copyCGImage(at: time, actualTime: nil)
//        let frameImg    : UIImage = UIImage(cgImage: img)
//        return frameImg
//    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let barHeight: CGFloat = UIApplication.shared.statusBarFrame.size.height
        let displayWidth: CGFloat = self.view.frame.width
        let displayHeight: CGFloat = self.view.frame.height
        
        myTableView = UITableView(frame: CGRect(x: 0, y: 0, width: displayWidth, height: displayHeight))
        myTableView.register(FramesTableViewCell.self, forCellReuseIdentifier: "FramesCell")
        myTableView.dataSource = self
        myTableView.delegate = self
        self.view.addSubview(myTableView)

        
//        var frameTime = 0.0
//        let durationTime = CMTimeGetSeconds(video.duration)
//        while frameTime < durationTime {
//            print(frameTime)
//            self.image = generateThumnail(url: videoURL, fromTime: frameTime)
//            self.imageCollection.append(self.image)
//            frameTime += 0.4
//        }
//
    }
    

    
}



extension FramesTableViewController: UITableViewDelegate, UITableViewDataSource {
    
    // MARK: - Table view data source
    
    func numberOfSections(in tableView: UITableView) -> Int {
        // #warning Incomplete implementation, return the number of sections
        return 1
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {

        return self.imageCollection.count
    }
    
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        return 300
    }
    
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "FramesCell", for: indexPath) as! FramesTableViewCell

        cell.frameImage.image = self.imageCollection[indexPath.row]
        
        
        return cell
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        tableView.deselectRow(at: indexPath, animated: true)
    }
    
    func tableView(_ tableView: UITableView, didHighlightRowAt indexPath: IndexPath) {
        
//        let leftOrRight = EyeImageData.instance.selectedLeftOrRight
//        if leftOrRight == "left" {
//
//            let image = self.imageCollection[indexPath.row]
//            let imageURL = save(imageName: "leftEye", image: image)
//
//
//        } else if leftOrRight == "right" {
//
//            let image = self.imageCollection[indexPath.row]
//           let imageURL = save(imageName: "rightEye", image: image)
//
//
//        }
        
        let image = self.imageCollection[indexPath.row]
        
    
        
//        //working implementation
//        let storyboard = UIStoryboard.init(name: "Main", bundle: nil)
//        let vc = storyboard.instantiateViewController(withIdentifier: "NewExamStoryboard") as! NewExamViewController
//        self.navigationController?.pushViewController(vc, animated: true)

        let cropViewController = CropViewController(image: image)
        cropViewController.rotateButtonsHidden = true
        cropViewController.hidesBottomBarWhenPushed = true
        cropViewController.resetAspectRatioEnabled = false
        cropViewController.rotateClockwiseButtonHidden = true
        cropViewController.aspectRatioPickerButtonHidden = true
        cropViewController.aspectRatioLockEnabled = true
        cropViewController.setAspectRatioPreset(CropViewControllerAspectRatioPreset.presetSquare, animated: false)
        cropViewController.delegate = self
        
        present(cropViewController, animated: true, completion: nil)
        
        
        
        
//        self.navigationController?.popToViewController(vc, animated: true)
//
        
//        self.performSegue(withIdentifier: "NewExamStoryboard", sender: self)
//        self.navigationController?.setViewControllers([vc], animated: true)

//        self.present(vc, animated: true, completion: nil)

        //self.navigationController?.popViewController(animated: false)
        //let _ = self.navigationController?.popToViewController((self.navigationController?.viewControllers[1]) as! NewExamViewController, animated: true)
        
        
        //let vc = storyboard.instantiateViewController(withIdentifier: "NewExamStoryboard") as! NewExamViewController
//        let navBar = UINavigationController.init(rootViewController: vc)
//
//        self.present(navBar, animated: true, completion: nil)
    }
    
}

extension FramesTableViewController: CropViewControllerDelegate {
    
    
    var documentsUrl: URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    
    private func save(imageName: String, image: UIImage) -> String? {
        let fileName = imageName
        let fileURL = documentsUrl.appendingPathComponent(fileName)
        if let imageData = UIImageJPEGRepresentation(image, 1.0) {
            try? imageData.write(to: fileURL, options: .atomic)
            return fileName // ----> Save fileName
        }
        print("Error saving image")
        return nil
    }
    
    
    
    func cropViewController(_ cropViewController: CropViewController, didCropToImage image: UIImage, withRect cropRect: CGRect, angle: Int) {
                //working implementation
                let leftOrRight = EyeImageData.instance.selectedLeftOrRight
                self.image = image
        if leftOrRight == "left" {
                save(imageName: "leftEye", image: image)
        } else if leftOrRight == "right" {
                save(imageName: "rightEye", image: image)
        }
        print("cropped the image")
        let storyboard = UIStoryboard.init(name: "Main", bundle: nil)
        let vc = storyboard.instantiateViewController(withIdentifier: "NewExamStoryboard") as! NewExamViewController
        self.navigationController?.pushViewController(vc, animated: true)
        dismiss(animated: true, completion: nil)
        //CropViewControllerAspectRatioPreset.presetSquare
        
    }
    
//    func cropViewController(_ cropViewController: CropViewController, didFinishCancelled cancelled: Bool) {
//        let storyboard = UIStoryboard.init(name: "Main", bundle: nil)
//        let vc = storyboard.instantiateViewController(withIdentifier: "NewExamStoryboard") as! NewExamViewController
//        self.navigationController?.pushViewController(vc, animated: true)
//    }

}
