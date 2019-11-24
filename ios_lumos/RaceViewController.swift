//
//  RaceViewController.swift
//  lumos ios
//
//  Created by Shalin Shah on 6/26/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit

class RaceViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource {
    
    @IBOutlet weak var tool: UIToolbar!
    @IBOutlet weak var nextButton: UIBarButtonItem!
    @IBOutlet weak var infoText: UILabel!
    
    @IBOutlet weak var pickerLabel: UIButton!
    @IBOutlet weak var pickerView: UIPickerView!
    @IBOutlet weak var raceLabel: UILabel!
    
    var races = ["White", "Hispanic / Latino", "Black / African American", "Native American", "Asian / Pacific Islander", "Other"]
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // make bottom toolbar taller
        tool.frame = CGRect(x: 0, y: view.frame.size.height - 55, width: view.frame.size.width, height: 55)
        
        
        
        // Text style for "NEXT"
        nextButton.title = "NEXT".uppercased()
        
        if let font = UIFont(name: "AvenirNext-DemiBold", size: 14) {
            nextButton.setTitleTextAttributes([NSAttributedStringKey.font:font], for: .normal)
        }
        
        nextButton.tintColor = UIColor(red: 1.00, green: 0.00, blue: 0.29, alpha: 1)
        
        
        // Text style for "Hi, I’m Lumos. What’s your name?"
        infoText.clipsToBounds = true
        infoText.alpha = 1
        
        infoText.text = "What's your nationality?"
        infoText.font = UIFont(name: "AvenirNext-Regular", size: 28)
        infoText.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        infoText.center.x = self.view.center.x
        
        
        pickerView.delegate = self
        pickerView.dataSource = self
        
        
        // Text style for "AGE"
        raceLabel.clipsToBounds = true
        raceLabel.alpha = 1
        raceLabel.text = "NATIONALITY".uppercased()
        raceLabel.font = UIFont(name: "AvenirNext-Regular", size: 13)
        //        ageLabel.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        raceLabel.textColor = UIColor(red: 84/255, green: 16/255, blue: 51/255, alpha: 1.0)
        
        
    }
    
    @IBAction func pickerLabelPressed(_ sender: Any) {
        if !pickerView.isHidden {
            pickerView.isHidden = true
        } else {
            pickerView.isHidden = false
        }
    }
    
    
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1;
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return races.count
    }
    
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return String(races[row])
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        
        pickerLabel.setBackgroundImage(#imageLiteral(resourceName: "select_white"), for: UIControlState.normal)
        raceLabel.textColor = UIColor(red: 1.00, green: 1.00, blue: 1.00, alpha: 1)
        pickerLabel.setTitleColor(UIColor.white, for: .normal)
        
        pickerLabel.setTitle("     " + String(races[row]), for: .normal)
        pickerView.isHidden = true
        
    }
    
    @IBAction func moveToHistory(_ sender: Any) {
        let storyBoard : UIStoryboard = UIStoryboard(name: "Main", bundle:nil)
        
        let resultViewController = storyBoard.instantiateViewController(withIdentifier: "HistoryView") as! HistoryViewController
        self.present(resultViewController, animated:true, completion:nil)
    }

    
}
