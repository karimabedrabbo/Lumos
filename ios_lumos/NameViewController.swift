//
//  NameViewController.swift
//  lumos ios
//
//  Created by Shalin Shah on 6/23/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit
import SkyFloatingLabelTextField

class NameViewController: UIViewController {

    @IBOutlet weak var tool: UIToolbar!
    @IBOutlet weak var nextButton: UIBarButtonItem!
    @IBOutlet weak var infoText: UILabel!
    
    
    var textField1: SkyFloatingLabelTextField!
    var textField2: SkyFloatingLabelTextField!

    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // make bottom toolbar taller
        tool.frame = CGRect(x: 0, y: view.frame.size.height - 55, width: view.frame.size.width, height: 55)
        

//        
//        let cancelBtn1 = UIButton(frame: CGRect(x: 0, y: 0, width: 20, height: 100))
//        cancelBtn1.setTitle("NEXT",for: .normal)
//        cancelBtn1.titleLabel!.font = UIFont(name: "AvenirNext-DemiBold", size: 14)
//        cancelBtn1.tintColor = UIColor(red: 1.00, green: 0.00, blue: 0.29, alpha: 1)
        
        
        

//        nextButton = UIBarButtonItem(customView: cancelBtn1)
        
        // Text style for "NEXT"
        nextButton.title = "NEXT".uppercased()


        
//        nextButton.setTitlePositionAdjustment(UIOffset(horizontal: 0, vertical: -1000), for: UIBarMetrics.default)
//        
//        nextButton.imageInsets = UIEdgeInsetsMake(20, 0, -40, 0)
//        
//        nextButton.titlePositionAdjustment(for: UIBarMetrics.default) = UIOffset(horizontal: 80, vertical: 80)
        



        
        if let font = UIFont(name: "AvenirNext-DemiBold", size: 14) {
            nextButton.setTitleTextAttributes([NSAttributedStringKey.font:font], for: .normal)
        }
        
        nextButton.tintColor = UIColor(red: 1.00, green: 0.00, blue: 0.29, alpha: 1)
        
        
        
        
        // Text style for "Hi, I’m Lumos. What’s your name?"
        infoText.clipsToBounds = true
        infoText.alpha = 1
        infoText.text = "Hi, I’m Lumos.\nWhat’s your name?"
        infoText.font = UIFont(name: "AvenirNext-Regular", size: 28)
        infoText.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        infoText.center.x = self.view.center.x
        
        
        
        
        // First and Last name input fields
        let lumosDarkColor = UIColor(red: 84/255, green: 16/255, blue: 51/255, alpha: 1.0)
        
        textField1 = SkyFloatingLabelTextField(frame: CGRect(x: self.view.center.x / 5, y: self.view.center.y - 100, width: view.frame.size.width - (2 * (self.view.center.x / 5)), height: 45))
        
//        print (view.center.x / 4)
        
        textField1.placeholder = "First"
        textField1.title = "First name"
        self.view.addSubview(textField1)
        
        textField2 = SkyFloatingLabelTextField(frame: CGRect(x: self.view.center.x / 5, y: self.view.center.y - 25, width: view.frame.size.width - (2 * (self.view.center.x / 5)), height: 45))
        textField2.placeholder = "Last"
        textField2.title = "Last name"
        
        
        // set all the colors
        textField1.tintColor = lumosDarkColor // the color of the blinking cursor
        textField1.placeholderColor = lumosDarkColor
        textField1.lineColor = lumosDarkColor
        textField1.textColor = UIColor.white
        textField1.selectedTitleColor = UIColor.white
        textField1.selectedLineColor = UIColor.white
        textField1.errorColor = UIColor.black
        textField2.tintColor = lumosDarkColor // the color of the blinking cursor
        textField2.placeholderColor = lumosDarkColor
        textField2.lineColor = lumosDarkColor
        textField2.textColor = UIColor.white
        textField2.selectedTitleColor = UIColor.white
        textField2.selectedLineColor = UIColor.white
        textField2.errorColor = UIColor.black
        
        // bottom line height in points
        textField1.lineHeight = 1.0
        textField1.selectedLineHeight = 2.0
        textField2.lineHeight = 1.0
        textField2.selectedLineHeight = 2.0

        self.view.addSubview(textField2)
        
        setDoneOnKeyboard()

        
//        
//        style.textColor = UIColor(red: 1.00, green: 0.00, blue: 0.29, alpha: 1)
        
        
    }
    
    func setDoneOnKeyboard() {
        let keyboardToolbar = UIToolbar()
        keyboardToolbar.sizeToFit()
        let flexBarButton = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        let doneBarButton = UIBarButtonItem(barButtonSystemItem: .done, target: self, action: #selector(dismissKeyboard))
        keyboardToolbar.items = [flexBarButton, doneBarButton]
        textField1.inputAccessoryView = keyboardToolbar
        textField2.inputAccessoryView = keyboardToolbar
    }
    
    @objc func dismissKeyboard() {
        if let text = textField1.text, !text.isEmpty
        {
            // change color to white
            textField1.lineColor = UIColor.white

        } else {
            // change color to purple
            textField1.lineColor = UIColor(red: 84/255, green: 16/255, blue: 51/255, alpha: 1.0)
        }

        if let text = textField2.text, !text.isEmpty
        {
            // change color to white
            textField2.lineColor = UIColor.white
        } else {
            // change color to lumos purple
            textField2.lineColor = UIColor(red: 84/255, green: 16/255, blue: 51/255, alpha: 1.0)
        }


        view.endEditing(true)
    }
    
    @IBAction func moveToAgeView(_ sender: Any) {
        let text = textField1.text, text2 = textField2.text
        
        if text!.characters.count < 2 || text!.characters.count > 12  {
            textField1.errorMessage = "Enter an actual name"
        } else if text2!.characters.count < 2 ||  text2!.characters.count > 13 {
            textField2.errorMessage = "Enter an actual name"
        } else {
            
            UserDefaults.standard.set(text, forKey: "firstName")
            UserDefaults.standard.set(text2, forKey: "lastName")
            
            let storyBoard : UIStoryboard = UIStoryboard(name: "Main", bundle:nil)
            
            let resultViewController = storyBoard.instantiateViewController(withIdentifier: "AgeView") as! AgeViewController
            self.present(resultViewController, animated:true, completion:nil)

        }
        

        
    }


    
    
}
