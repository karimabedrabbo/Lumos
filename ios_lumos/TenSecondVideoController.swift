//
//  TenSecondVideoController.swift
//  lumos ios
//
//  Created by Johnathan Chen on 2/20/18.
//  Copyright © 2018 Shalin Shah. All rights reserved.
//

import Foundation
import SwiftyCam
import UIKit
import AVFoundation


class TenSecondVideoController: SwiftyCamViewController {

    var maxDuration : Double = 10
    var captureButton: UIButton!
    var cancelButton: UIButton!
    var progressTimer: Timer?
    var progress : Double = 0
//    var timeLabel: UILabel!
    
    var imageCollection = [UIImage]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Setting the camera delegate
        cameraDelegate = self
        
        // Setting maximum duration for video
        maximumVideoDuration = maxDuration
        shouldUseDeviceOrientation = true
        allowAutoRotate = false
        addButtons()
        
    }

    override var prefersStatusBarHidden: Bool {
        return true
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        captureButton.isSelected = true
    }
    
    override func viewWillAppear(_ animated: Bool) {
        
        //Hiding navigation bar to allow the Camera view to be full screen
        self.navigationController?.isNavigationBarHidden = true
    }
    
    // MARK: - Action Functions (Buttons)
    
    // Function which controls the camera switch button
    @objc private func cameraSwitchAction(_ sender: Any) {
//        switchCamera()
    }
    
    
    // Function which controls the cancel button
    @objc private func cancel()
    {
        
        self.navigationController?.isNavigationBarHidden = false
        self.navigationController?.popViewController(animated: true)
        
    }

    
    @objc func update()
    {
        progress += 0.2
        if progress < maxDuration {
            takePhoto()
        }
        else {
            stop()
        }

    }
    
    // Adding buttons programatically to the Camera view
    private func addButtons() {
        captureButton = UIButton(frame: CGRect(x: view.frame.midX - 37.5, y: view.frame.height - 100.0, width: 70.0, height: 70.0))
        let image = UIImage(named: "capture_photo")
        captureButton.setImage(image, for: UIControlState.normal)
        
        captureButton.addTarget(self, action: #selector(record), for: .touchDown)
        captureButton.addTarget(self, action: #selector(stop), for: UIControlEvents.touchUpInside)
        self.view.addSubview(captureButton)
        //captureButton.delegate = self
        
        cancelButton = UIButton(frame: CGRect(x: 20.0, y: 30.0, width: 20.0, height: 20.0))
        cancelButton.setImage(#imageLiteral(resourceName: "cancel"), for: UIControlState())
        cancelButton.addTarget(self, action: #selector(cancel), for: .touchUpInside)
        view.addSubview(cancelButton)
        
    }
    
    @objc func record() {

        
        UIView.animate(withDuration: 0.25, animations: {
            self.cancelButton.alpha = 0.0
        })
        //counts 10 seconds
        update()
        self.progressTimer = Timer.scheduledTimer(timeInterval: 0.2, target: self, selector: #selector(update), userInfo: nil, repeats: true)
        //        startVideoRecording()
//        print("Started recording")
//        updateProgress()
//
//        let width = CGFloat(200)
//        timeLabel = UILabel(frame: CGRect(x: self.view.center.x, y: self.view.center.y, width: width, height: 21))
//        timeLabel.center = CGPoint(x: self.view.center.x, y: self.view.center.y - (self.view.center.y * 0.5))
//        timeLabel.textAlignment = .center
//        timeLabel.font = UIFont(name: "AvenirNext-DemiBold", size: 25)
//        timeLabel.textColor = UIColor.white
//        timeLabel.text = String(self.progress)
//        self.view.addSubview(timeLabel)
    }
    
    
    @objc func stop() {
   
        UIView.animate(withDuration: 0.25, animations: {
            self.cancelButton.alpha = 1.0
        })
            progress = 0.0
            progressTimer?.invalidate()
            let newVC = FramesTableViewController(imagesTaken: self.imageCollection)
            self.navigationController?.pushViewController(newVC, animated: true)
        
        
//        timeLabel.removeFromSuperview()
        //stopVideoRecording()
        //print("Stopped recording")
        
    }
    
    
//    func updateProgress() {
//
//         //counts 10 seconds
//        self.progressTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { (timer) in
//            self.progress += 1
//            //self.timeLabel.text = String(self.progress)
//            print("progress:", self.progress)
//
//        }
    
        
    }
    



// MARK: - SwiftyCamViewControllerDelegate
extension TenSecondVideoController : SwiftyCamViewControllerDelegate
{
    
    
//    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didChangeZoomLevel zoom: CGFloat) {
//        print(zoom)
//    }
//
//    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didSwitchCameras camera: SwiftyCamViewController.CameraSelection) {
//        print(camera)
//    }
//
//    //Functin called when startVideoRecording() is called
//    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didBeginRecordingVideo camera: SwiftyCamViewController.CameraSelection) {
//        //print("Did Begin Recording")
//        // captureButton.growButton()
//        UIView.animate(withDuration: 0.25, animations: {
//            self.cancelButton.alpha = 0.0
//        })
//    }
//
//    // Function called when stopVideoRecording() is called
//    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didFinishRecordingVideo camera: SwiftyCamViewController.CameraSelection) {
//
////        self.progressTimer?.invalidate()
////        UIView.animate(withDuration: 0.25, animations: {
////            self.cancelButton.alpha = 1.0
////        })
//    }
    
    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didTake photo: UIImage) {
        imageCollection.append(photo)
    }
//    // Function called once recorded has stopped. The URL for the video gets returned here.
//    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didFinishProcessVideoAt url: URL) {
//        
//        // I am passing the url to FramesTableViewController to show captured frames
//        let newVC = FramesTableViewController(videoURL: url, progress: progress - 1.0)
//        self.navigationController?.pushViewController(newVC, animated: true)
//         progressTimer?.invalidate()
////        self.present(newVC, animated: true, completion: nil)
//
//    }
    
    // Function which allows you to zoom. Added animation for User/UI purposes
    func swiftyCam(_ swiftyCam: SwiftyCamViewController, didFocusAtPoint point: CGPoint) {
        let focusView = UIImageView(image: #imageLiteral(resourceName: "focus"))
        focusView.center = point
        focusView.alpha = 0.0
        view.addSubview(focusView)
        
        UIView.animate(withDuration: 0.25, delay: 0.0, options: .curveEaseInOut, animations: {
            focusView.alpha = 1.0
            focusView.transform = CGAffineTransform(scaleX: 1.25, y: 1.25)
        }, completion: { (success) in
            UIView.animate(withDuration: 0.15, delay: 0.5, options: .curveEaseInOut, animations: {
                focusView.alpha = 0.0
                focusView.transform = CGAffineTransform(translationX: 0.6, y: 0.6)
            }, completion: { (success) in
                focusView.removeFromSuperview()
            })
        })
    }
}

