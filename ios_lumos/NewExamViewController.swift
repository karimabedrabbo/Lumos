//
//  NewExamViewController.swift
//  lumos ios
//
//  Created by Shalin Shah on 6/27/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit
import Alamofire
import SwiftyJSON
import AlamofireImage
import Async
//import Fusuma

class NewExamViewController: UIViewController {

    @IBOutlet weak var headingLabel: UILabel!
    
    var timeLabel: UILabel!
    var whichEye: String = ""
    var leftImage : UIImage?
    var rightImage : UIImage?
    var eyeImages = [UIImage]()
    @IBOutlet weak var leftEyeImg: UIImageView!
    @IBOutlet weak var rightEyeImg: UIImageView!
    
    var leftImageHM : UIImage!
    var rightImageHM : UIImage!
    
    
    func setup() {
        // set up nav bar
        self.navigationController?.navigationBar.topItem?.title = " "
        navigationController?.navigationBar.tintColor = .white
        self.navigationController?.navigationBar.titleTextAttributes = [ NSAttributedStringKey.font: UIFont(name: "AvenirNext-DemiBold", size: 15)!, NSAttributedStringKey.foregroundColor : UIColor.white]
        self.navigationItem.title = "AM I OKAY?".uppercased()
        
        
        // Text style for "TAKE A PICTURE" Label
        headingLabel.clipsToBounds = true
        headingLabel.alpha = 1
        headingLabel.text = "TAKE A PICTURE".uppercased()
        headingLabel.font = UIFont(name: "AvenirNext-Regular", size: 11)
        headingLabel.textColor = UIColor(red: 0.64, green: 0.64, blue: 0.64, alpha: 1)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setup()
        
        // load left eye image
        if let loadLeftImage = load(fileName: "leftEye") {
            leftEyeImg.image = loadLeftImage
        }
        
        // load right eye image
        if let loadRightImage = load(fileName: "rightEye") {
            rightEyeImg.image = loadRightImage
        }


    }
    
    @IBAction func leftEyeButtonPressed(_ sender: AnyObject) {
        
        EyeImageData.instance.selectedLeftOrRight = "left"
        whichEye = "left"
        
        let vc = TenSecondVideoController()
        self.navigationController?.pushViewController(vc, animated: true)
    }
    
    @IBAction func rightEyeButtonPressed(_ sender: AnyObject) {
        
        EyeImageData.instance.selectedLeftOrRight = "right"
        whichEye = "right"
        
        let vc = TenSecondVideoController()
        self.navigationController?.pushViewController(vc, animated: true)
    }
    
    func saveImage (image: UIImage, path: String ) -> URL{
        let data = UIImagePNGRepresentation(image)
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let filename = paths[0].appendingPathComponent(path)
        try? data!.write(to: filename)
        return filename
    }
    func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        let size = image.size
        
        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        
        // Figure out what our orientation is, and use that to form the rectangle
        var newSize: CGSize
        if(widthRatio > heightRatio) {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }
        
        // This is the rect that we've calculated out and this is what is actually used below
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        
        // Actually do the resizing to the rect using the ImageContext stuff
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }

    
    @IBAction func startScreenPressed(_ sender: UIButton) {

        showHUD()
        
        // request
        let leftName = "left.png"
        let newImageOne = resizeImage(image: leftEyeImg.image!, targetSize: CGSize(width: 128, height: 128))
        let leftImagePath = saveImage(image: newImageOne, path: leftName)
        print(leftImagePath)
        
        Alamofire.upload(
            multipartFormData: { multipartFormData in
                multipartFormData.append(leftImagePath, withName: "upload1")
        },
            //                print(multipartFormData)
            to: "http://app.lumosvision.com/get_heatmap",
            encodingCompletion: { encodingResult in
                switch encodingResult {
                case .success(let upload, _, _):
                    print("succcess!!!")
                    upload.responseImage { response in
                        self.leftImageHM = response.result.value
                        print("image downloaded: \(self.leftImageHM)")
                        self.getRightHM()
                    }
                case .failure(let encodingError):
                    print(encodingError)
                }
        }
        )
        
        
}
    
func getRightHM() {
    let rightName = "right.png"
    let newImageTwo = resizeImage(image: rightEyeImg.image!, targetSize: CGSize(width: 128, height: 128))
    let rightImagePath = saveImage(image: newImageTwo, path: rightName)
    print(rightImagePath)
        
    Alamofire.upload(
        multipartFormData: { multipartFormData in
            multipartFormData.append(rightImagePath, withName: "upload1")
    },
        to: "http://app.lumosvision.com/get_heatmap",
        encodingCompletion: { encodingResult in
            switch encodingResult {
            case .success(let upload, _, _):
                print("succcess!!!")
                upload.responseImage { response in
                    self.rightImageHM = response.result.value
                    print("image downloaded: \(self.rightImageHM)")
                    self.genReport()
                }
            case .failure(let encodingError):
                print(encodingError)
            }
    }
    )
}

func genReport() {
    // request
    let leftName = "left.png"
    let newImageOne = resizeImage(image: leftEyeImg.image!, targetSize: CGSize(width: 128, height: 128))
    let leftImagePath = saveImage(image: newImageOne, path: leftName)
    print(leftImagePath)
    
    
    let rightName = "right.png"
    let newImageTwo = resizeImage(image: rightEyeImg.image!, targetSize: CGSize(width: 128, height: 128))
    let rightImagePath = saveImage(image: newImageTwo, path: rightName)
    print(rightImagePath)
    
    Alamofire.upload(
        multipartFormData: { multipartFormData in
            multipartFormData.append(leftImagePath, withName: "upload1")
            multipartFormData.append(rightImagePath, withName: "upload2")
    },
        //                print(multipartFormData)
        to: "http://app.lumosvision.com/upload",
        encodingCompletion: { encodingResult in
            switch encodingResult {
            case .success(let upload, _, _):
                upload.responseJSON { response in
                    debugPrint(response)
                    
                    if let value = response.result.value {
                        let report = JSON(value)
                        print("JSON: \(report)")
                        //                            finalReport = report
                        //                            print(response)
                        
                        self.hideHUD()
                        let vc = self.storyboard?.instantiateViewController(withIdentifier: "ResultsView") as! ResultsViewController
                        vc.report = report
                        vc.leftImageHM = self.rightImageHM
                        vc.rightImageHM = self.leftImageHM
                        self.navigationController?.pushViewController(vc, animated: true)
                    }
                }
            case .failure(let encodingError):
                print(encodingError)
            }
            
            print("uploaded")
    }
    )
    }
}

//
//func createRequestBodyWith(parameters:[String:NSObject], filePathKey:String, boundary:String) -> NSData{
//    let body = NSMutableData()
//
//    for (key, value) in parameters {
//        body.appendString(string: "--\(boundary)\r\n")
//        body.appendString(string: "Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
//        body.appendString(string: "\(value)\r\n")
//    }
//
//    body.appendString(string: "--\(boundary)\r\n")
//
//    var mimetype = "image/jpg"
//
//    let defFileName = "yourImageName.jpg"
//
//    let imageData = UIImageJPEGRepresentation(yourImage, 1)
//
//    body.appendString(string: "Content-Disposition: form-data; name=\"\(filePathKey!)\"; filename=\"\(defFileName)\"\r\n")
//    body.appendString(string: "Content-Type: \(mimetype)\r\n\r\n")
//    body.append(imageData!)
//    body.appendString(string: "\r\n")
//
//    body.appendString(string: "--\(boundary)--\r\n")
//
//    return body
//}
//
//func generateBoundaryString() -> String {
//    return "Boundary-\(NSUUID().uuidString)"
//}
//
//extension NSMutableData {
//    func appendString(string: String) {
//        let data = string.data(using: String.Encoding.utf8, allowLossyConversion: true)
//        append(data!)
//    }
//}


let hudView = UIView(frame: CGRect(x: 0, y: 0, width: 80, height: 80))
let indicatorView = UIActivityIndicatorView(frame: CGRect(x: 0, y: 0, width: 80, height: 80))
extension NewExamViewController {
    
    var documentsUrl: URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    
    private func load(fileName: String) -> UIImage? {
        let fileURL = documentsUrl.appendingPathComponent(fileName)
        do {
            let imageData = try Data(contentsOf: fileURL)
            return UIImage(data: imageData)
        } catch {
            print("Error loading image : \(error)")
        }
        return nil
    }
    
    func showHUD() {
        hudView.center = CGPoint(x: view.frame.size.width/2, y: view.frame.size.height/2)
        hudView.backgroundColor = UIColor.darkGray
        hudView.alpha = 0.9
        hudView.layer.cornerRadius = hudView.bounds.size.width/2
        
        indicatorView.center = CGPoint(x: hudView.frame.size.width/2, y: hudView.frame.size.height/2)
        indicatorView.activityIndicatorViewStyle = UIActivityIndicatorViewStyle.white
        hudView.addSubview(indicatorView)
        indicatorView.startAnimating()
        view.addSubview(hudView)
    }
    func hideHUD() { hudView.removeFromSuperview() }
}


