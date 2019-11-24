//
//  HomeViewController.swift
//  
//
//  Created by Shalin Shah on 6/27/17.
//
//

import UIKit

class HomeViewController: UIViewController, UITableViewDelegate, UITableViewDataSource{
    
    let dates  = ["Mar 11, 2018", "Mar 11, 2018", "Mar 10, 2018", "Mar 9, 2018"]

    @IBOutlet weak var tableHeadingLabel: UILabel!
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    public func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return dates.count
    }

    public func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "Cell") as! HomeTableViewCell
        
        cell.selectionStyle = .none


//        the following commented code increases cell border on all sides of the cell
//        cell.layer.borderWidth = 15.0
//        cell.layer.borderColor = UIColor.white.cgColor
//
        
        // the following code increases cell border only on specified borders
        let bottom_border = CALayer()
        let bottom_padding = CGFloat(10.0)
        bottom_border.borderColor = UIColor.white.cgColor
        bottom_border.frame = CGRect(x: 0, y: cell.frame.size.height - bottom_padding, width:  cell.frame.size.width, height: cell.frame.size.height)
        bottom_border.borderWidth = bottom_padding
        
        let right_border = CALayer()
        let right_padding = CGFloat(15.0)
        right_border.borderColor = UIColor.white.cgColor
        right_border.frame = CGRect(x: cell.frame.size.width - right_padding, y: 0, width: right_padding, height: cell.frame.size.height)
        right_border.borderWidth = right_padding
        
        let left_border = CALayer()
        let left_padding = CGFloat(15.0)
        left_border.borderColor = UIColor.white.cgColor
        left_border.frame = CGRect(x: 0, y: 0, width: left_padding, height: cell.frame.size.height)
        left_border.borderWidth = left_padding
        
//        let top_border = CALayer()
//        let top_padding = CGFloat(10.0)
//        top_border.borderColor = UIColor.white.cgColor
//        top_border.frame = CGRect(x: 0, y: 0, width: cell.frame.size.width, height: top_padding)
//        top_border.borderWidth = top_padding
        
        
        cell.layer.addSublayer(bottom_border)
        cell.layer.addSublayer(right_border)
        cell.layer.addSublayer(left_border)
//        cell.layer.addSublayer(top_border)


        cell.layer.masksToBounds = true
        
        
        cell.dateLabel.text = dates[indexPath.row]
        cell.examNumberLabel.text = ("EXAM " + toWords(number: (indexPath.row + 1))!).uppercased()
        
       
        return cell
    }
    override func viewDidLoad() {
        super.viewDidLoad()
        
        

        navigationController?.navigationBar.barTintColor = hexStringToUIColor(hex: "ff014a")

        let imageView = UIImageView(frame: CGRect(x: 0, y: 0, width: 50, height: 20))
        let image = #imageLiteral(resourceName: "white_logo")
        imageView.contentMode = .scaleAspectFit // set imageview's content mode
        imageView.image = image
        self.navigationItem.titleView = imageView
        
        // Settings button
        let button = UIButton.init(type: .custom)
        button.setImage(#imageLiteral(resourceName: "settings_icon"), for: UIControlState.normal)
        button.frame = CGRect.init(x: 0, y: 0, width: 25, height: 25) //CGRectMake(0, 0, 30, 30)
        let barButton = UIBarButtonItem.init(customView: button)
        self.navigationItem.leftBarButtonItem = barButton
        
        // Text style for "PREVIOUS RESULTS" Label
        tableHeadingLabel.clipsToBounds = true
        tableHeadingLabel.alpha = 1
        tableHeadingLabel.text = "PREVIOUS RESULTS".uppercased()
        tableHeadingLabel.font = UIFont(name: "AvenirNext-Regular", size: 11)
        tableHeadingLabel.textColor = UIColor(red: 0.64, green: 0.64, blue: 0.64, alpha: 1)

        


    }
    
    
    
    func toWords<N>(number: N) -> String? {
        let formatter = NumberFormatter()
        formatter.numberStyle = .spellOut
        
        switch number {
        case is Int, is UInt, is Float, is Double:
            return formatter.string(from: number as! NSNumber)
        case is String:
            if let number = Double(number as! String) {
                return formatter.string(from: NSNumber(floatLiteral: number))
            }
        default:
            break
        }
        return nil
    }
    
    func hexStringToUIColor (hex:String) -> UIColor {
        var cString:String = hex.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        
        if (cString.hasPrefix("#")) {
            cString.remove(at: cString.startIndex)
        }
        
        if ((cString.characters.count) != 6) {
            return UIColor.gray
        }
        
        var rgbValue:UInt32 = 0
        Scanner(string: cString).scanHexInt32(&rgbValue)
        
        return UIColor(
            red: CGFloat((rgbValue & 0xFF0000) >> 16) / 255.0,
            green: CGFloat((rgbValue & 0x00FF00) >> 8) / 255.0,
            blue: CGFloat(rgbValue & 0x0000FF) / 255.0,
            alpha: CGFloat(1.0)
        )
    }

    
    
}
