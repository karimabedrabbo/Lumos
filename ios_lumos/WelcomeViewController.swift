//
//  WelcomeViewController.swift
//  lumos ios
//
//  Created by Shalin Shah on 6/23/17.
//  Copyright © 2017 Shalin Shah. All rights reserved.
//

import UIKit

class WelcomeViewController: UIViewController {
    @IBOutlet weak var welcome: UILabel!
    @IBOutlet weak var descrip: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        // Text style for "WELCOME TO"
        welcome.clipsToBounds = true
        welcome.alpha = 1
        welcome.text = "WELCOME TO".uppercased()
        welcome.font = UIFont(name: "AvenirNext-DemiBold", size: 15)
        welcome.textColor = UIColor(red: 1.00, green: 0.00, blue: 0.29, alpha: 1)
        welcome.center.x = self.view.center.x
        
        
        // Text style for "Automated screening for glaucoma"
        descrip.clipsToBounds = true
        descrip.alpha = 1
        descrip.text = "Automated screening\nfor glaucoma"
        
        descrip.font = UIFont(name: "AvenirNext-Regular", size: 20)
        descrip.textColor = UIColor(red: 0.24, green: 0.24, blue: 0.24, alpha: 1)
        descrip.center.x = self.view.center.x

        
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

